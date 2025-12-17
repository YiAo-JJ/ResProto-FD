import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from .distortions import *

def get_transform_waterbirds(aug_types, disturb_type=None, disturb_level=0):
    transform_list = []
    
    # distortions（Level 1-5）
    param_dict = {
        'gaussian_noise': [0.001, 0.002, 0.005, 0.01, 0.05],
        'block_wise': [16, 32, 48, 64, 80],
        'color_contrast': [0.85, 0.725, 0.6, 0.475, 0.35],
        'color_saturation': [0.4, 0.3, 0.2, 0.1, 0.0],
        'jpeg_compression': [2, 3, 4, 5, 6],
    }
    if disturb_type is not None and disturb_type in param_dict:
        level = max(1, min(disturb_level, 5))
        disturb_param = param_dict[disturb_type][level - 1]

        def apply_disturb(img):
            img = transforms.ToPILImage()(img)  # tensor -> PIL
            img = np.array(img)  # PIL -> numpy array (HWC)

            if disturb_type == 'gaussian_noise':
                img = gaussian_noise_color(img, disturb_param)
                print('Disturb: gaussian_noise')
            elif disturb_type == 'block_wise':
                img = block_wise(img, disturb_param)
                print('Disturb: block_wise')
            elif disturb_type == 'color_contrast':
                img = color_contrast(img, disturb_param)
                print('Disturb: color_contrast')
            elif disturb_type == 'color_saturation':
                img = color_saturation(img, disturb_param)
                print('Disturb: color_saturation')
            elif disturb_type == 'jpeg_compression':
                img = jpeg_compression(img, disturb_param)
                print('Disturb: jpeg_compression')

            return transforms.ToTensor()(img)  # numpy -> tensor

        transform_list.append(transforms.Lambda(apply_disturb))
        transform_list.append(transforms.Resize((224, 224)))
    
    else:
        transform_list = []
        transform_list.append(transforms.ToTensor())
        if 'space' in aug_types:
            transform_list.append(transforms.Resize((300, 300)))
            transform_list.append(transforms.RandomChoice([
                transforms.RandomCrop((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            ]))
            print('Aug: space')
        else:
            transform_list.append(transforms.Resize((224, 224)))
        
        if 'texture' in aug_types:
            transform_list.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.5)),
            ], p=0.3))
            print('Aug: texture')
        
        if 'mask' in aug_types:
            transform_list.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
            print('Aug: mask')
        
    transform_list.append(transforms.Normalize(mean=FFpp._normalization_stats['mean'],
                            std=FFpp._normalization_stats['std']))
        
    transform = transforms.Compose(transform_list)
    return transform

class FFpp(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}
    def __init__(self, split: str, transform: transforms, root: str = "../DataSets/", 
                 manipu_types: list = ["Deepfakes"], compress: list = ['c23', 'c23'], aug_types: str = 'None',
                 disturb_type: str = 'None', disturb_level: int = 0,):
        self.split = split
        self.aug_types = aug_types
        if transform is None:
            self.transform = get_transform_waterbirds(self.aug_types, disturb_type, disturb_level)
        else:
            self.transform = transform
        self.root = root
        self.manipu_types = manipu_types
        self.compress = compress
        self.classnums = [0,0]

        self.classnames = ['real', 'forged']

        # Load JSON
        json_path = os.path.join(root, f"{self.split}.json")
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)

        self.data = []
        self.video = []
        self.labels = []
        self.spec_labels = []

        # original_sequences
        original_path = os.path.join(root, "original_sequences", "youtube", self.compress[0], "frames")
        mask_path = os.path.join(root, "original_sequences", "youtube", self.compress[0], "masks")
        for pair in self.pairs:
            for id in pair:
                id_path = os.path.join(original_path, id)
                if os.path.exists(id_path):
                    for img_name in os.listdir(id_path):
                        mask_name = os.path.join(mask_path, id, img_name)
                        if os.path.exists(mask_name):
                            self.video.append(id)
                            self.data.append(os.path.join(id_path, img_name))
                            self.labels.append(0)  # Real 0
                            self.classnums[0] +=1

        # manipulated_sequences
        for mtp in self.manipu_types:
            manipulated_path = os.path.join(root, "manipulated_sequences", mtp, self.compress[1], "frames")
            for pair in self.pairs:
                id1, id2 = pair
                # add id1_id2 和 id2_id1
                for id_pair in [f"{id1}_{id2}", f"{id2}_{id1}"]:
                    id_path = os.path.join(manipulated_path, id_pair)
                    if os.path.exists(id_path):
                        for img_name in os.listdir(id_path):
                            self.video.append(id_path)
                            self.data.append(os.path.join(id_path, img_name))
                            self.labels.append(1)  # Forged 1
                            self.classnums[1] +=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name = self.video[idx]
        image_path = self.data[idx]
        label = self.labels[idx]

        # Load Image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load Transform
        if self.transform:
            image = self.transform(image)
        item = {
            "video_name": video_name,
            "image_path": image_path,
            "image": image,
            "label": label,
        }
        return item


if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize([256, 256]),
                                        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ])

    manipu_types = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    train_dataset = FFpp(
        split="train",
        transform=None,
        root="../DataSets/FaceForensics++/",
        manipu_types=manipu_types,
        compress=["c23", "c23"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )


    test_dataset = FFpp(
        split="test",
        transform=None,
        root="../DataSets/FaceForensics++/",
        manipu_types=manipu_types,
        compress=["c23", "c23"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    for idx,data in enumerate(train_dataloader):
        print(data["image"].shape)
        print(data["label"])
        print(len(train_dataloader))
        break
    for idx,data in enumerate(test_dataloader):
        print(data["image"].shape)
        print(data["label"])
        print(len(test_dataloader))
        break