import torch
import torch.nn as nn
import torch.nn.functional as F
from heapq import heappush, heappop
import torch.distributed as dist
from sklearn.cluster import KMeans

def cluster(image_features, num_clusters = 2):
    feats_np = image_features.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(feats_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=image_features.dtype, device=image_features.device)
    return centers

def get_prototypes_cls(image_features, labels, cluster_param=4):
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)

    image_features_0 = torch.tensor(0.0, device=labels.device)
    image_features_1 = torch.tensor(0.0, device=labels.device)
    # average
    if mask_0.any():
        image_features_0 = torch.mean(image_features[mask_0], dim=0).unsqueeze(0)
    
    # cluster
    if mask_1.any():
        if image_features[mask_1].size(0) > cluster_param-1:
            image_features_1 = cluster(image_features[mask_1], cluster_param)
    return [image_features_0, image_features_1]

class GRP(nn.Module):
    """
    Gradient-aware Residual Prototypes
    """
    def __init__(self, feat_dim: int = 512, num_size: int = 50, temperature = 0.1, top_k=64, cluster_param=4):
        super(GRP, self).__init__()
        self.k_per_class = num_size // 2
        self.feat_dim = feat_dim
        self.prototype_set  = {0: torch.empty(0, feat_dim).cuda(), 1: torch.empty(0, feat_dim).cuda()}
        self.labels= {0: [], 1: []}
        self.scores = {0: torch.empty(0).cuda(), 1: torch.empty(0).cuda()}
        self.step_count = {0: torch.empty(0).cuda(), 1: torch.empty(0).cuda()}
        self.temperature = temperature
        self.top_k = top_k
        self.cluster_param = cluster_param

    @torch.no_grad()
    def update_batch_smpl_cluster(self, image_features: torch.Tensor, labels: torch.Tensor, grads_cls, gama=0.99):
        prototypes_cls = get_prototypes_cls(image_features, labels, cluster_param=self.cluster_param)
        
        cls = 0
        score = grads_cls[cls]
        prototype = prototypes_cls[cls]
        if prototype.shape != torch.Size([]):
            self.step_count[cls] += 1
            self.scores[cls] = self.scores[cls] * gama
            # —— Not full: insert directly ——
            if len(self.prototype_set[cls]) < self.k_per_class:
                self.prototype_set[cls] = torch.cat([self.prototype_set[cls], prototype], dim=0)
                self.labels[cls].append(cls)
                self.scores[cls] = torch.cat([self.scores[cls], torch.tensor([score]).cuda()], dim=0)  # Update score
                self.step_count[cls] = torch.cat([self.step_count[cls], torch.ones([1]).cuda()], dim=0)

            # —— Full: Compare to the top of the heap ——
            else:
                scores_tmp = self.scores[cls] * (gama ** self.step_count[cls])
                heap_tmp = []
                for i in range(len(scores_tmp)):
                    heappush(heap_tmp, (scores_tmp[i].item(), i))
                worst_score, worst_idx = heap_tmp[0]   # Minimum score
                if score > worst_score:
                    self.prototype_set[cls][worst_idx] = prototype
                    self.labels[cls][worst_idx] = cls
                    self.scores[cls][worst_idx] = score  # Update score
                    self.step_count[cls][worst_idx] = 1
                # Discard
                del heap_tmp

        cls = 1
        score = grads_cls[cls]
        prototype = prototypes_cls[cls]
        if prototype.shape != torch.Size([]):
            self.step_count[cls] += 1
            self.scores[cls] = self.scores[cls] * gama
            # —— Not full: insert directly ——
            if len(self.prototype_set[cls]) < self.k_per_class:
                self.prototype_set[cls] = torch.cat([self.prototype_set[cls], prototype], dim=0)
                self.labels[cls].extend([cls]*prototype.size(0))
                self.scores[cls] = torch.cat([self.scores[cls], torch.tensor([score]*prototype.size(0)).cuda()], dim=0)  # Update score
                self.step_count[cls] = torch.cat([self.step_count[cls], torch.ones([prototype.size(0)]).cuda()], dim=0)

            # —— Full: Compare to the top of the heap ——
            else:
                scores_tmp = self.scores[cls] * (gama ** self.step_count[cls])
                heap_tmp = []
                for i in range(len(scores_tmp)):
                    heappush(heap_tmp, (scores_tmp[i].item(), i))
                for i_fea in range(prototype.size(0)):
                    worst_score, worst_idx = heap_tmp[0]   # Minimum score
                    if score > worst_score:
                        self.prototype_set[cls][worst_idx] = prototype[i_fea]
                        self.labels[cls][worst_idx] = cls
                        self.scores[cls][worst_idx] = score  # Update score
                        self.step_count[cls][worst_idx] = 1
                        heappop(heap_tmp)
                        heappush(heap_tmp, (score, worst_idx))
                # Discard
                del heap_tmp

    @torch.no_grad()
    def sync_pts(self, gama=0.99):
        """Synchronize prototype sets data of all processes"""
        for cls in [0, 1]:
            # Collect data about the current process
            local_pts = self.prototype_set[cls].detach()
            local_scores = self.scores[cls].detach()
            local_steps = self.step_count[cls].detach()
            
            # Get data for all processes
            world_size = dist.get_world_size()
            all_pts = [torch.empty_like(local_pts) for _ in range(world_size)]
            all_scores = [torch.empty_like(local_scores) for _ in range(world_size)]
            all_steps = [torch.empty_like(local_steps) for _ in range(world_size)]
            
            dist.all_gather(all_pts, local_pts)
            dist.all_gather(all_scores, local_scores)
            dist.all_gather(all_steps, local_steps)
            
            # Merge global data
            global_pts = torch.cat(all_pts, dim=0)
            global_scores = torch.cat(all_scores, dim=0)
            global_steps = torch.cat(all_steps, dim=0)
            scores_tmp = global_scores* (gama ** global_steps)
            
            # Select the k_per_class samples with the highest scores
            if len(global_scores) > self.k_per_class:
                _, topk_indices = torch.topk(scores_tmp, self.k_per_class, largest=True)
                selected_pts = global_pts[topk_indices]
                selected_scores = global_scores[topk_indices]
                selected_steps = global_steps[topk_indices]
            else:
                selected_pts = global_pts
                selected_scores = global_scores
                selected_steps = global_steps
            
            # Update the prototype set of the current process
            self.prototype_set[cls] = selected_pts
            self.scores[cls] = selected_scores
            self.labels[cls] = [cls] * len(selected_pts)
            self.step_count[cls] = selected_steps

    def forward(self, x):
        all_prototypes = []
        pts_labels = []
        for key in sorted(self.prototype_set.keys()):
            all_prototypes.append(self.prototype_set[key])
            pts_labels.extend([key]*len(self.prototype_set[key]))
        
        all_prototypes = torch.cat(all_prototypes, dim=0).cuda()
        pts_labels = torch.tensor(pts_labels).cuda()

        dist = torch.mm(F.normalize(x), all_prototypes.transpose(1, 0))    # B * Q, prototypes already normalized
        # compute cos similarity between each feature vector and prototypes ---> [B, N]
        sim_weight, sim_indices = torch.topk(dist, k=self.k)
        sim_labels = torch.gather(pts_labels.expand(x.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = F.softmax(sim_weight / self.temperature, dim=1)

        one_hot_label = torch.zeros(x.size(0) * self.k, 2, device=sim_labels.device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, 2) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores