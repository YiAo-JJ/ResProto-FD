import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def get_transform_waterbirds():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop([224, 224]),
        transforms.Normalize(mean=DFD._normalization_stats['mean'],
                             std=DFD._normalization_stats['std']),
    ])
    return transform

class DFD(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}
    def __init__(self, split: str, transform: transforms, root: str = "../DataSets/FaceForensics++",):
        self.split = split
        if transform is None:
            self.transform = get_transform_waterbirds()
        else:
            self.transform = transform
        self.root = root

        self.data = []
        self.video = []
        self.labels = []
        self.classnames = ['real', 'forged']
        self.classnums = [0,0]

        real_path = os.path.join(self.root, 'original_sequences', 'actors', 'c23', 'frames')
        fake_path = os.path.join(self.root, 'manipulated_sequences', 'DeepFakeDetection', 'c23', 'frames')

        folder_path=''
        for video_name in os.listdir(real_path):
            folder_path = os.path.join(real_path, video_name)
            for image_name in os.listdir(folder_path):
                self.video.append(folder_path)
                self.data.append(os.path.join(folder_path, image_name))
                self.labels.append(0)  # Real 0
                self.classnums[0] +=1

        folder_path=''
        for video_name in os.listdir(fake_path):
            folder_path = os.path.join(fake_path, video_name)
            for image_name in os.listdir(folder_path):
                self.video.append(folder_path)
                self.data.append(os.path.join(folder_path, image_name))
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
        
        # Transform
        if self.transform:
            image = self.transform(image)
        item = {
            "video_name": video_name,
            "image_path": image_path,
            "image": image,
            "label": label
        }
        return item


if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize([256, 256]),
                                        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ])
    test_dataset = DFD(
        split="test",
        transform=transform,
        root="../DataSets/FaceForensics++/",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    for idx,data in enumerate(test_dataloader):
        print(data["image"].shape)
        print(data["label"])
        print(len(test_dataloader))
        break