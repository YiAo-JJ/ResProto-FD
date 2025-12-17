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
        transforms.Normalize(mean=DF40._normalization_stats['mean'],
                             std=DF40._normalization_stats['std']),
    ])
    return transform

class DF40(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}
    def __init__(self, split: str, transform: transforms, root: str = "../DataSets/DF40",
                 manipu_types: list = ["e4s"],):
        self.split = split
        if transform is None:
            self.transform = get_transform_waterbirds()
        else:
            self.transform = transform
        self.root = root
        self.manipu_types = manipu_types
        self.ff_root = os.path.join(root, 'FaceForensics++')
        self.save_root = os.path.join(root, 'Celeb-DF-v2')
        self.save_video_list = []
        self.classnames = ['real', 'forged']

        self.data = []
        self.labels = []
        self.classnums = [0,0,0,0]

        for mtp in manipu_types:
            save_json_path = os.path.join(root, "dataset_json", f'{mtp}_save.json')
            with open(save_json_path, 'r') as f:
                save_data_pairs = json.load(f)

            real_path = save_data_pairs[f'{mtp}_save'][f'{mtp}_Real']['test']
            for id in real_path.keys():
                for frame_path in real_path[id]['frames']:
                    image_path = ''
                    if 'deepfakes_detection_datasets/DF40' in frame_path:
                        image_path = frame_path.replace('deepfakes_detection_datasets/DF40', self.root)
                    else:
                        image_path = frame_path.replace('deepfakes_detection_datasets', self.root)
                    self.data.append(image_path)
                    self.labels.append(0)  # Real 0
                    self.classnums[0]+=1

            fake_path = save_data_pairs[f'{mtp}_save'][f'{mtp}_Fake']['test']
            for id in fake_path.keys():
                for frame_path in fake_path[id]['frames']:
                    image_path = ''
                    if 'deepfakes_detection_datasets/DF40' in frame_path:
                        image_path = frame_path.replace('deepfakes_detection_datasets/DF40', self.root)
                    else:
                        image_path = frame_path.replace('deepfakes_detection_datasets', self.root)
                    self.data.append(image_path)
                    self.labels.append(1)  # Forged 1
                    self.classnums[1]+=1


            ff_json_path = os.path.join(root, "dataset_json", f'{mtp}_ff.json')
            with open(ff_json_path, 'r') as f:
                ff_data_pairs = json.load(f)

            real_path = ff_data_pairs[f'{mtp}_ff'][f'{mtp}_Real']['test']
            for id in real_path.keys():
                for frame_path in real_path[id]['frames']:
                    image_path = ''
                    if 'deepfakes_detection_datasets/DF40' in frame_path:
                        image_path = frame_path.replace('deepfakes_detection_datasets/DF40', self.root)
                    else:
                        image_path = frame_path.replace('deepfakes_detection_datasets', self.root)
                    self.data.append(image_path)
                    self.labels.append(0)  # Real 0
                    self.classnums[2]+=1

            fake_path = ff_data_pairs[f'{mtp}_ff'][f'{mtp}_Fake']['test']
            for id in fake_path.keys():
                for frame_path in fake_path[id]['frames']:
                    image_path = ''
                    if 'deepfakes_detection_datasets/DF40' in frame_path:
                        image_path = frame_path.replace('deepfakes_detection_datasets/DF40', self.root)
                    else:
                        image_path = frame_path.replace('deepfakes_detection_datasets', self.root)
                    self.data.append(image_path)
                    self.labels.append(1)  # Forged 1
                    self.classnums[3]+=1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]

        # Load Image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Transform
        if self.transform:
            image = self.transform(image)
        item = {
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
    test_dataset = DF40(
        split="test",
        transform=None,
        root="../DataSets/DF40/",
        manipu_types=['e4s']
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    print(len(test_dataloader))
    print(test_dataset.classnums)
