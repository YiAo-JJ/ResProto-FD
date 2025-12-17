import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from .clip.clip import tokenize, load
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from .prompt_templates import fake_templates, real_templates
from .RL import *
from .GRP import GRP
import copy
from collections import defaultdict
from sklearn.cluster import KMeans

def cluster(image_features, num_clusters = 2):
    feats_np = image_features.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(feats_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=image_features.dtype, device=image_features.device)
    return centers

def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())

def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

def comput_metrics(labels_list, scores):

    fpr, tpr, thresholds = ROC_AUC(labels_list, scores)
    auroc = AUC_score(fpr=fpr, tpr=tpr)
    ap = average_precision_score(labels_list, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return auroc, ap, eer

def compute_time(start_time, end_time):
    t = int(end_time-start_time)
    str_time = ""
    seconds = 0
    minutes = 0
    hours = 0
    if t >= 60:
        seconds = t % 60
        minutes = t // 60
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            str_time = str(f"{hours}h{minutes}m{seconds}s")
        else:
            str_time = str(f"{minutes}m{seconds}s")
    else:
        seconds = t
        str_time = str(f"{seconds}s")
    return str_time


class CLIPModel(nn.Module):

    def __init__(self, clip_name, class_names, templates, cuda_id=0, cpu=False, device="cuda:0", res_lambda=0.3):
        super(CLIPModel, self).__init__()
        self.cuda_id = cuda_id
        if cpu:
            self.clip, _ = load(clip_name, device="cpu")
        else:
            self.clip, _ = load(clip_name, f'cuda:{self.cuda_id}')
        self.feature_dim = self.clip.visual.output_dim
        self.device = device
        self.class_names = class_names
        self.prompt_templates = templates
        self.res_lambda = res_lambda

    def _init_text_features(self):
        """Initialize fixed text features"""
        text_features = []
        if self.prompt_templates is None:
            real_tokens = tokenize(real_templates).to(next(self.clip.parameters()).device)
            real_features = self.clip.encode_text(real_tokens)
            fake_tokens = tokenize(fake_templates).to(next(self.clip.parameters()).device)
            fake_features = self.clip.encode_text(fake_tokens)
            text_features.append(real_features.mean(dim=0))
            text_features.append(fake_features.mean(dim=0))
            return torch.stack(text_features)
        else:
            prompts = [[temp.format(c) for temp in self.prompt_templates] for c in self.class_names]
            for cls_prompts in prompts:
                tokens = tokenize(cls_prompts).to(next(self.clip.parameters()).device)
                features = self.clip.encode_text(tokens)
                text_features.append(features.mean(dim=0))
            return torch.stack(text_features)
      
    def forward(self, input, labels=None):
        # get the image features
        image_features = self.clip.encode_image(input)
        text_features = self._init_text_features()

        # normalized features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = self.clip.logit_scale.exp()
        if labels is not None:
            lam = self.res_lambda
            image_features = image_features - lam * text_features[labels]
        similarity = logit_scale * image_features @ text_features.t()
        return similarity, image_features

class CLIPTrainer:
    def __init__(self, train_sampler, train_dataloader, test_dataloader, clip_path, classes, templates=["a photo of a {} face."],
                 device='cpu', best_save_dir='', classnums=None, res_lambda=0.3, pts_num=128, cluster_param=4, gama=0.99):
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.res_lambda = res_lambda
        self.pts_num = pts_num
        self.gama = gama
        self.classnums = torch.Tensor(classnums).to(self.device)
        self.model = CLIPModel(clip_path, classes, templates, device=device, res_lambda=res_lambda).to(self.device)
        top_k = pts_num // 2
        self.prototype_sets = GRP(feat_dim=self.model.feature_dim, num_size=pts_num, temperature=0.1, top_k=top_k, cluster_param=cluster_param)
        # Train ImageEncoder and freeze TextEncoder
        for name, param in self.model.named_parameters():
            if 'clip.visual' in name:
                param.requires_grad_(True)
            elif 'clip.logit_scale' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.best_save_dir = best_save_dir
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=5e-7,
            weight_decay=1e-6
        )
        self.scaler = torch.amp.GradScaler('cuda')
    
    def compute_gradient_contribution_batch(self, loss_fun, images, labels, use_Residual=False):
        """
        :param loss_fn: ResidualLearningLoss after instantiation
        """
        model_copy = copy.deepcopy(self.model).cuda()
        optimizer_copy = copy.deepcopy(self.optimizer)
        model_params = list(model_copy.parameters())
        for group in optimizer_copy.param_groups:
            group['params'] = model_params
        scaler_copy = copy.deepcopy(self.scaler)
        loss_fun_copy = copy.deepcopy(loss_fun)

        model_copy.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            if use_Residual:
                outputs_copy, image_features_copy = model_copy(images, labels)
            else:
                outputs_copy, image_features_copy = model_copy(images)

        loss_cls = loss_fun_copy.forward_percls(outputs_copy, labels)  # [B, 2]
        optimizer_copy.zero_grad(set_to_none=True)

        grads_cls0 = []
        scaler_copy.scale(loss_cls[0]).backward(retain_graph=True)
        for p in model_copy.parameters():
            if p.grad is not None:
                grads_cls0.append(p.grad.detach().clone().flatten())
        grad_vec_cls0 = torch.cat(grads_cls0)
        grad_norm_cls0 = torch.norm(grad_vec_cls0).item()
        optimizer_copy.zero_grad(set_to_none=True)

        grads_cls1 = []
        scaler_copy.scale(loss_cls[1]).backward()
        for p in model_copy.parameters():
            if p.grad is not None:
                grads_cls1.append(p.grad.detach().clone().flatten())
        grad_vec_cls1 = torch.cat(grads_cls1)
        grad_norm_cls1 = torch.norm(grad_vec_cls1).item()
        optimizer_copy.zero_grad(set_to_none=True)

        del model_copy, optimizer_copy, scaler_copy
        return [grad_norm_cls0, grad_norm_cls1], image_features_copy

    def train(self, max_epoch=20, use_Residual=False, use_RL=True, use_GRP=True, start_epoch=5):
        if use_RL:
            loss_fun = ResidualLearningLoss(cls_num_list=self.classnums)
        else:
            loss_fun = torch.nn.CrossEntropyLoss()
        model_nums = max(0, (len(os.listdir(self.best_save_dir)))//4)
        save_nums = model_nums
        self.model.train()
        epoch_loss = []
        results = {'auroc':[]}
        for epoch in range(0, max_epoch):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_start_time = time.time()
            for step, data in enumerate(self.train_dataloader):
                images = data['image'].contiguous().cuda(non_blocking=True)
                labels = data['label'].contiguous().cuda(non_blocking=True)
            
                total_loss = 0
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    if use_Residual:
                        outputs, _ = self.model(images, labels)
                    else:
                        outputs, _ = self.model(images)

                if use_GRP:
                    # -------- Update Prototype sets --------
                    if epoch >= start_epoch:
                        grad_cls,image_features_copy = self.compute_gradient_contribution_batch(loss_fun, images.detach(), labels, use_Residual)
                        self.prototype_sets.update_batch_smpl_cluster(image_features_copy.detach(), labels, grad_cls, self.gama)

                total_loss = loss_fun(outputs, labels)
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss.append(total_loss.item())
                print("\rTraining: Epoch: {0}\{1} Batch: {2}\{3} | loss: {4},".format(epoch+1, max_epoch, step+1, len(self.train_dataloader), total_loss.item()), end="")
                # if step > 64:
                #     break

            epoch_end_time = time.time()
            epoch_time = compute_time(epoch_start_time, epoch_end_time)
            print(f" epoch_loss: {np.mean(epoch_loss)}, epoch_time: {epoch_time}")

            test_start_time = time.time()
            if epoch >= start_epoch:
                all_labels, all_scores, auc, ap, eer = self.test(use_GRP=use_GRP)
            else:
                all_labels, all_scores, auc, ap, eer = self.test()
            os.makedirs(os.path.join(self.best_save_dir), exist_ok=True)

            results['auroc'].append(auc)
            if auc==max(results['auroc']):
                best_model_dict = self.model.state_dict()
                best_grp_dict = self.prototype_sets
                best_scores = all_scores
                best_labels = all_labels
                torch.save(best_model_dict, os.path.join(self.best_save_dir, f"model-v{save_nums}"))
                torch.save(best_scores, os.path.join(self.best_save_dir, f"scores-v{save_nums}.tsr"))
                torch.save(best_labels, os.path.join(self.best_save_dir, f"labels-v{save_nums}.tsr"))
                save_nums += 1
            if epoch>=start_epoch:
                best_grp_dict = self.prototype_sets
                os.makedirs(os.path.join(self.best_save_dir, 'GRP'), exist_ok=True)
                if use_GRP:
                    torch.save(best_grp_dict, os.path.join(self.best_save_dir, 'GRP', f"pts-ep{epoch+1}"))

            test_end_time = time.time()
            test_time = compute_time(test_start_time, test_end_time)
            print(f' auc: {auc}, ap: {ap}, eer: {eer}', end='')
            print(f', test_time: {test_time}\n')

            with open(os.path.join(self.best_save_dir, "../results.txt"), 'a') as file:
                file.writelines(f"epoch: {epoch+1}, auroc: {auc}, ap: {ap}, eer: {eer}")
                file.writelines(f'\n')

    def test(self, use_GRP=False):
        self.model.eval()
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for step, data in enumerate(self.test_dataloader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                with torch.amp.autocast('cuda'):
                    outputs, image_features = self.model(images)

                    all_labels.append(labels)
                    if use_GRP:
                        all_scores.append(self.prototype_sets(image_features)[:,-1])
                    else:
                        all_scores.append(outputs[:,-1])

                    print("\rTesting: {0}/{1}".format(step+1, len(self.test_dataloader)), end="")
        auc, ap, eer = comput_metrics(all_labels, all_scores)
        return all_labels, all_scores, auc, ap, eer


    def test_video(self, use_GRP=False):
        self.model.eval()
        all_labels, all_scores = [], []
        all_video_names = []

        with torch.no_grad():
            for step, data in enumerate(self.test_dataloader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                video_names = data['video_name']  # list[str]

                with torch.amp.autocast('cuda'):
                    outputs, image_features = self.model(images)

                    all_labels.append(labels)
                    if use_GRP:
                        all_scores.append(self.prototype_sets(image_features)[:,-1])
                    else:
                        all_scores.append(outputs[:,-1])

                    all_video_names.extend(video_names)
                    print(f"\rTesting: {step + 1}/{len(self.test_dataloader)}", end="")

        # Aggregate frame-level results
        all_labels = torch.cat(all_labels, dim=0).cpu()
        all_scores = torch.cat(all_scores, dim=0).cpu()

        # Constructing a video-level dictionary
        video_scores_dict = defaultdict(list)
        video_labels_dict = {}

        for name, score, label in zip(all_video_names, all_scores, all_labels):
            video_scores_dict[name].append(score.item())
            video_labels_dict[name] = label.item()

        # Video-level aggregation (optional mean/max)
        video_labels, video_scores = [], []
        for video in video_scores_dict:
            video_labels.append(video_labels_dict[video])
            video_scores.append(sum(video_scores_dict[video]) / len(video_scores_dict[video]))  # mean

        video_labels = torch.tensor(video_labels).numpy()
        video_scores = torch.tensor(video_scores).numpy()

        auc_video, ap_video, eer_video = comput_metrics(video_labels, video_scores)

        return video_labels, video_scores, auc_video, ap_video, eer_video


    def train_ddp(self, max_epoch=20, use_Residual=False, use_RL=True, use_GRP=True, start_epoch=5):
        if use_RL:
            loss_fun = ResidualLearningLoss(cls_num_list=self.classnums)
        else:
            loss_fun = torch.nn.CrossEntropyLoss()
        model_nums = max(0, (len(os.listdir(self.best_save_dir)))//4)
        save_nums = model_nums
        self.model.train()
        epoch_loss = []
        results = {'auroc':[]}
        local_rank = int(os.environ["LOCAL_RANK"])
        for epoch in range(0, max_epoch):
            self.train_sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            epoch_start_time = time.time()
            for step, data in enumerate(self.train_dataloader):
                images = data['image'].contiguous().cuda(non_blocking=True)
                labels = data['label'].contiguous().cuda(non_blocking=True)
            
                total_loss = 0
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    if use_Residual:
                        outputs, _ = self.model(images, labels)
                    else:
                        outputs, _ = self.model(images)

                if use_GRP:
                    # -------- Update Prototype sets --------
                    if epoch >= start_epoch:
                        grad_cls,image_features_copy = self.compute_gradient_contribution_batch(loss_fun, images.detach(), labels, use_Residual)
                        self.prototype_sets.update_batch_smpl_cluster(image_features_copy.detach(), labels, grad_cls, self.gama)

                total_loss = loss_fun(outputs, labels)
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss.append(total_loss.item())
                if local_rank==0:
                    print("\rTraining: Epoch: {0}\{1} Batch: {2}\{3} | loss: {4},".format(epoch+1, max_epoch, step+1, len(self.train_dataloader), total_loss.item()), end="")
                # if step > 64:
                #     break
            
            if use_GRP:
                # -------- Synchronous Prototype sets --------
                self.prototype_sets.sync_pts()
            if local_rank==0:
                epoch_end_time = time.time()
                epoch_time = compute_time(epoch_start_time, epoch_end_time)
                print(f" epoch_loss: {np.mean(epoch_loss)}, epoch_time: {epoch_time}")

            test_start_time = time.time()
            if epoch >= start_epoch:
                all_labels, all_scores, auc, ap, eer = self.test_ddp(local_rank, use_GRP=use_GRP)
            else:
                all_labels, all_scores, auc, ap, eer = self.test_ddp(local_rank)
            os.makedirs(os.path.join(self.best_save_dir), exist_ok=True)

            if local_rank == 0:
                results['auroc'].append(auc)
                if auc==max(results['auroc']):
                    best_model_dict = self.model.state_dict()
                    best_grp_dict = self.prototype_sets
                    best_scores = all_scores
                    best_labels = all_labels
                    torch.save(best_model_dict, os.path.join(self.best_save_dir, f"model-v{save_nums}"))
                    torch.save(best_scores, os.path.join(self.best_save_dir, f"scores-v{save_nums}.tsr"))
                    torch.save(best_labels, os.path.join(self.best_save_dir, f"labels-v{save_nums}.tsr"))
                    save_nums += 1
                if epoch>=start_epoch:
                    best_grp_dict = self.prototype_sets
                    os.makedirs(os.path.join(self.best_save_dir, 'GRP'), exist_ok=True)
                    if use_GRP:
                        torch.save(best_grp_dict, os.path.join(self.best_save_dir, 'GRP', f"pts-ep{epoch+1}"))

                test_end_time = time.time()
                test_time = compute_time(test_start_time, test_end_time)
                print(f' auc: {auc}, ap: {ap}, eer: {eer}', end='')
                print(f', test_time: {test_time}\n')

                with open(os.path.join(self.best_save_dir, "../results.txt"), 'a') as file:
                    file.writelines(f"epoch: {epoch+1}, auroc: {auc}, ap: {ap}, eer: {eer}")
                    file.writelines(f'\n')

    def test_ddp(self, local_rank, use_GRP=False):
        self.model.eval()
        local_labels = []
        local_scores = []

        with torch.no_grad():
            for step, data in enumerate(self.test_dataloader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                with torch.amp.autocast('cuda'):
                    outputs, image_features = self.model(images)

                    local_labels.append(labels)
                    if use_GRP:
                        local_scores.append(self.prototype_sets(image_features)[:,-1])
                    else:
                        local_scores.append(outputs[:,-1])

                    if local_rank==0:
                        print("\rTesting: {0}/{1}".format(step+1, len(self.test_dataloader)), end="")
        local_labels = torch.cat(local_labels, dim=0)
        local_scores = torch.cat(local_scores, dim=0)
        all_labels = [torch.zeros_like(local_labels) for _ in range(2)]
        all_scores = [torch.zeros_like(local_scores) for _ in range(2)]
        torch.distributed.all_gather(all_labels, local_labels)
        torch.distributed.all_gather(all_scores, local_scores)
        all_labels = torch.cat(all_labels, dim=0).cpu()
        all_scores = torch.cat(all_scores, dim=0).cpu()
        auc, ap, eer = comput_metrics(all_labels, all_scores)
        return all_labels, all_scores, auc, ap, eer

    def test_video_ddp(self, local_rank, use_GRP=False):
        self.model.eval()
        local_videos, local_labels, local_scores = [], [], []
        local_video_names = []

        with torch.no_grad():
            for step, data in enumerate(self.test_dataloader):
                images = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                video_names = data['video_name']  # list[str]
                with torch.amp.autocast('cuda'):
                    outputs, image_features = self.model(images)

                if use_GRP:
                    scores = self.prototype_sets(image_features)[:, -1]
                else:
                    scores = outputs[:, -1]

                local_labels.append(labels)
                local_scores.append(scores)
                local_video_names.extend(video_names)
                if local_rank == 0:
                    print(f"\rTesting: {step + 1}/{len(self.test_dataloader)}", end="")

        # Aggregate frame-level results
        local_labels = torch.cat(local_labels, dim=0)
        local_scores = torch.cat(local_scores, dim=0)

        all_labels = [torch.zeros_like(local_labels) for _ in range(2)]
        all_scores = [torch.zeros_like(local_scores) for _ in range(2)]
        all_video_names = [None for _ in range(2)]

        torch.distributed.all_gather(all_labels, local_labels)
        torch.distributed.all_gather(all_scores, local_scores)

        # Aggregated video name
        gathered_names = [None for _ in range(2)]
        torch.distributed.all_gather_object(gathered_names, local_video_names)

        all_labels = torch.cat(all_labels, dim=0).cpu()
        all_scores = torch.cat(all_scores, dim=0).cpu()
        all_video_names = sum(gathered_names, [])

        # Constructing a video-level dictionary
        video_scores_dict = defaultdict(list)
        video_labels_dict = {}

        for name, score, label in zip(all_video_names, all_scores, all_labels):
            video_scores_dict[name].append(score.item())
            video_labels_dict[name] = label.item()

        # Video-level aggregation (optional mean/max)
        video_labels, video_scores = [], []
        for video in video_scores_dict:
            video_labels.append(video_labels_dict[video])
            video_scores.append(sum(video_scores_dict[video]) / len(video_scores_dict[video]))  # mean

        video_labels = torch.tensor(video_labels).numpy()
        video_scores = torch.tensor(video_scores).numpy()

        auc_video, ap_video, eer_video = comput_metrics(video_labels, video_scores)
        return video_labels, video_scores, auc_video, ap_video, eer_video

if __name__ == '__main__':
    image = torch.randn((32,3,224,224)).cuda()
    clip_model = CLIPModel()
    clip_model(image)