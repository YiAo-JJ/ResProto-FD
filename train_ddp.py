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
    cmdline_parser = argparse.ArgumentParser('Train ResProto-FD')
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
    cmdline_parser.add_argument('--aug_param',
                                default= 'space-mask-texture',
                                type=str,
                                help='space-mask-texture')
    cmdline_parser.add_argument('--max_epoch',
                                default= 20,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--use_Residual',
                                default= True,
                                type=str2bool,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--use_RL',
                                default= True,
                                type=str2bool,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--use_GRP',
                                default= True,
                                type=str2bool,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--start_epoch',
                                default= 2,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--cluster_param',
                                default=4,
                                type=int,
                                help='Load checkpoint')
    cmdline_parser.add_argument('--gama',
                                default=0.99,
                                type=float,
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
    aug_types = args.aug_param
    if args.train_dataset == "FF++" or args.train_dataset == "FaceForensics++":
        train_dataset = FFpp(
            split="train",
            transform=None,
            root="../../DataSets/FaceForensics++/",
            manipu_types=manipu_types,
            compress=["c23", "c23"],
            aug_types=aug_types,
        )
    if args.test_dataset == "FF++" or args.test_dataset == "FaceForensics++":
        test_dataset = FFpp(
            split="test",
            transform=None,
            root="../../DataSets/FaceForensics++/",
            manipu_types=manipu_types,
            compress=["c23", "c23"],
        )
    elif 'Celeb-DF' in args.test_dataset:
        test_dataset = CelebDF(
            split="test",
            transform=None,
            root=f"../../DataSets/{args.test_dataset}/",
        )
    elif args.test_dataset == "DFDC":
        test_dataset = DFDC(
            split="test",
            transform=None,
            root=f"../../DataSets/{args.test_dataset}/",
        )
    elif args.test_dataset == "DFD":
        test_dataset = DFD(
            split="test",
            transform=None,
            root=f"../../DataSets/FaceForensics++/",
        )
    elif args.test_dataset == "DF40":
        test_dataset = DF40(
            split="test",
            transform=None,
            root=f"../../DataSets/DF40/",
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
    test_save_dir = os.path.join('results', args.train_dataset, args.results_dir, args.test_dataset)
    if args.test_dataset == 'DF40':
        test_save_dir = os.path.join('results', args.train_dataset, args.results_dir, args.test_dataset, manipu_types[0])
    os.makedirs(test_save_dir, exist_ok=True)
    # Instantiation
    trainer = CLIPTrainer(
        train_sampler=train_sampler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        clip_path=args.clip_path,
        classes=train_dataset.classnames,
        templates=None,
        device=device,
        best_save_dir=best_save_dir,
        classnums=train_dataset.classnums,
        res_lambda = args.res_lambda,
        pts_num = args.pts_num,
        cluster_param = args.cluster_param,
        gama = args.gama,
    )
    trainer.model = DDP(trainer.model.module if hasattr(trainer.model, 'module') else trainer.model, device_ids=[local_rank], find_unused_parameters=True)

    # Training
    trainer.train_ddp(max_epoch=args.max_epoch, use_Residual=args.use_Residual, use_RL=args.use_RL, use_GRP=args.use_GRP, start_epoch=args.start_epoch)
    dist.destroy_process_group()