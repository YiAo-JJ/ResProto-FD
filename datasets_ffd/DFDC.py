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
        transforms.Normalize(mean=DFDC._normalization_stats['mean'],
                             std=DFDC._normalization_stats['std']),
    ])
    return transform

class DFDC(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}
    def __init__(self, split: str, transform: transforms, root: str = "../DataSets/DFDC",):
        self.split = split
        if transform is None:
            self.transform = get_transform_waterbirds()
        else:
            self.transform = transform
        self.root = root
        self.data_json = dict
        self.video_list = []
        self.classnames = ['real', 'forged']

        # Load json
        try:
            json_path = os.path.join(root, self.split, f"metadata.json")
            with open(json_path, 'r') as f:
                self.data_json = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'{self.split} dataset not exist!')

        self.video_list = self.data_json.keys()

        self.data = []
        self.video = []
        self.labels = []
        self.classnums = [0,0]

        for video_name in self.video_list:
            video_path = video_name.split('.')[0]

            folder_path = os.path.join(self.root, self.split, 'frames', video_path)

            if os.path.exists(folder_path):
                image_list = os.listdir(folder_path)

                for image_name in image_list:
                    self.video.append(folder_path)
                    self.data.append(os.path.join(folder_path, image_name))
                    self.labels.append(self.data_json[video_name]['is_fake'])  # Real 0 Forged 1
                    self.classnums[int(self.data_json[video_name]['is_fake'])] +=1

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
    test_dataset = DFDC(
        split="test",
        transform=transform,
        root="../DataSets/DFDC/",
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