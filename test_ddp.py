import os
os.environ['NCCL_TIMEOUT'] = '3600'
import numpy as np
import torch
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from src import *
from datasets_ffd import *
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve

def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())

def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    cmdline_parser = argparse.ArgumentParser('Test ResProto-FD')
    cmdline_parser.add_argument('--clip_path',
                                default= 'ViT-B/16',
                                help='RN50,ViT-B/16')
    cmdline_parser.add_argument('--train_dataset',
                                default= "FaceForensics++",
                                help='Load checkpoint')
    cmdline_parser.add_argument('--test_dataset',
                                default= "FaceForensics++",
                                help='Load checkpoint')
    cmdline_parser.add_argument('--manipu_type',
                                default= "Deepfakes,Face2Face,FaceSwap,NeuralTextures",
                                help= "(FF++)Deepfakes,Face2Face,FaceSwap,NeuralTextures,\
                                        (DF40)uniface,e4s,facedancer,fsgan,inswap,simswap",)
    cmdline_parser.add_argument('--train_batch_size',
                                default= 32,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--test_batch_size',
                                default= 128,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--num_workers',
                                default= 8,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--results_dir',
                                default= 'ViTB16-ResProto-FD',
                                type=str,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--res_lambda',
                                default= 0.3,
                                type=float,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--pts_num',
                                default= 128,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--use_GRP',
                                default= True,
                                type=str2bool,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--test_level',
                                default= 'frame',
                                type=str,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--top_k',
                                default= 100,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--disturb_type',
                                default= None,
                                type=str,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--disturb_level',
                                default= 0,
                                type=int,
                                help='Load checkpoint')
    args, unknowns = cmdline_parser.parse_known_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    train_dataset=""
    test_dataset=""
    manipu_types = args.manipu_type.split(',')
    if args.test_dataset == "FF++" or args.test_dataset == "FaceForensics++":
        test_dataset = FFpp(
            split="test",
            transform=None,
            root="../DataSets/FaceForensics++/",
            manipu_types=manipu_types,
            compress=["c23", "c23"],
            disturb_type=args.disturb_type,
            disturb_level=args.disturb_level,
        )
    elif 'Celeb-DF' in args.test_dataset:
        test_dataset = CelebDF(
            split="test",
            transform=None,
            root=f"../DataSets/{args.test_dataset}/",
        )
    elif args.test_dataset == "DFDC":
        test_dataset = DFDC(
            split="test",
            transform=None,
            root=f"../DataSets/{args.test_dataset}/",
        )
    elif args.test_dataset == "DFD":
        test_dataset = DFD(
            split="test",
            transform=None,
            root=f"../DataSets/FaceForensics++/",
        )
    elif args.test_dataset == "DF40":
        test_dataset = DF40(
            split="test",
            transform=None,
            root=f"../DataSets/DF40/",
            manipu_types=manipu_types,
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        # shuffle=True,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2,
    )

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        # shuffle=False,
        sampler=test_sampler,
        persistent_workers=True,
    )

    best_save_dir = os.path.join('results', args.train_dataset, args.results_dir, "best")
    os.makedirs(best_save_dir, exist_ok=True)
    # Instantiation
    trainer = CLIPTrainer(
        train_sampler=None,
        train_dataloader=None,
        test_dataloader=test_dataloader,
        clip_path=args.clip_path,
        classes=test_dataset.classnames,
        templates=None,
        device=device,
        best_save_dir=best_save_dir,
        classnums=test_dataset.classnums,
        res_lambda = args.res_lambda,
        pts_num = args.pts_num,
    )
    trainer.model = DDP(trainer.model.module if hasattr(trainer.model, 'module') else trainer.model, device_ids=[local_rank], find_unused_parameters=True)

    # Evaluation
    model_name_list = [n for n in os.listdir(best_save_dir) if "model" in n]
    model_name_list.sort()
    model_state = torch.load(os.path.join(best_save_dir, model_name_list[-1]), weights_only=True)
    trainer.model.load_state_dict(model_state)

    if args.use_GRP:
        GRP_folder = os.path.join(best_save_dir, 'GRP')
        GRP_state = torch.load(os.path.join(GRP_folder, 'pts-ep8'))
        trainer.prototype_sets = GRP_state
        trainer.prototype_sets.top_k = args.top_k

    if args.test_level == 'frame':
        test_save_dir = os.path.join('results', args.train_dataset, args.results_dir, args.test_dataset)
        if args.disturb_type is not None:
            test_save_dir = os.path.join('results', args.train_dataset, args.results_dir, f"{args.test_dataset}_{args.disturb_type}")
        if args.test_dataset == 'DF40':
            test_save_dir = os.path.join(test_save_dir, manipu_types[0])
        os.makedirs(test_save_dir, exist_ok=True)
        all_labels, all_scores, auroc, ap, eer = trainer.test_ddp(local_rank, use_GRP=args.use_GRP)
        lbl_num = int((len(os.listdir(test_save_dir)))//2)
        if local_rank==0:
            print(f" Test auroc: {auroc:.2%}, ap: {ap:.2%}, eer: {eer:.2%}")

            with open(os.path.join(test_save_dir, "results.txt"), 'a') as file:
                if args.disturb_type is not None:
                    file.writelines(f"top_k-{args.top_k}\t{args.disturb_type}-{args.disturb_level}\tauroc: {auroc}, ap: {ap}, eer: {eer}.\n")
                else:
                    file.writelines(f"top_k-{args.top_k}\tauroc: {auroc}, ap: {ap}, eer: {eer}.\n")
            torch.save(all_labels, os.path.join(test_save_dir, f"labels{lbl_num}.tsr"))
            torch.save(all_scores, os.path.join(test_save_dir, f"scores{lbl_num}.tsr"))
    if args.test_level == 'video':
        video_labels, video_scores, auc_video, ap_video, eer_video = trainer.test_video_ddp(local_rank, use_GRP=args.use_GRP)
        test_video_save_dir = os.path.join('results', args.train_dataset, args.results_dir, f"{args.test_dataset}_video")
        if args.disturb_type is not None:
            test_video_save_dir = os.path.join('results', args.train_dataset, args.results_dir, f"{args.test_dataset}_video_{args.disturb_type}")
        os.makedirs(test_video_save_dir, exist_ok=True)
        lbl_num = int((len(os.listdir(test_video_save_dir)))//2)
        if local_rank==0:
            print(f" Test video auroc: {auc_video:.2%}, ap: {ap_video:.2%}, eer: {eer_video:.2%}")

            with open(os.path.join(test_video_save_dir, "results.txt"), 'a') as file:
                if args.disturb_type is not None:
                    file.writelines(f"top_k-{args.top_k}\t{args.disturb_type}-{args.disturb_level}\tauroc: {auc_video}, ap: {ap_video}, eer: {eer_video}.\n")
                else:
                    file.writelines(f"top_k-{args.top_k}\tauroc: {auc_video}, ap: {ap_video}, eer: {eer_video}.\n")
            torch.save(video_labels, os.path.join(test_video_save_dir, f"video_labels{lbl_num}.tsr"))
            torch.save(video_scores, os.path.join(test_video_save_dir, f"video_scores{lbl_num}.tsr"))
    dist.destroy_process_group()