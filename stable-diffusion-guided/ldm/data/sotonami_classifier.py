import io
import random
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image 

PATH = "/lfs/skampere1/0/pura/datasets/sotonami/"

def preprocess_image(filename, interpolation, transforms):
    image = Image.open(filename).convert("RGB")
    assert image.size[0] == image.size[1]
    size, _ = image.size
    image = image.resize((256, 256), interpolation)
    image = transforms(image)
    image = np.array(image).astype(np.uint8)
    return (image / 127.5 - 1.0).astype(np.float32)

class IshidaSuiClassifierDataset(Dataset):
    def __init__(self, size=None, interpolation="bicubic", split=0.9, seed=25, **kwargs):
        self.interpolation = {"linear": Image.BILINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]

        # loading positives 
        df = pd.read_csv(PATH + "metadata.csv")
        pos = pd.DataFrame()
        pos["filename"] = df["filename"].map(lambda x: f"data/{x}")
        pos["label"] = 1

        # loading negatives
        neg = pd.DataFrame({
            'filename': [f"control/samples/{i:05d}.png" for i in range(n + 1)], 
            'label': [0] * (n + 1) 
        })

        self.data = pd.concat([pos, neg], ignore_index=True)

        # Get train-val split
        random.seed(seed)
        n_train = int(len(self.data) * split)
        split = ["train"] * n_train + ["valid"] * (len(self.data) - n_train)
        random.shuffle(split)
        self.data["split"] = pd.Series(split)

        # Data augmentation
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),        
            transforms.RandomRotation(15),  
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
        self.val_transforms = self.train_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data["split"][idx] == "train":
            transforms = self.train_transforms
        else:
            transforms = self.val_transforms
        path = PATH + self.data["file_name"][idx]
        return {
            "image" : preprocess_image(path, self.interpolation, transforms),
            "caption" : self.data["text"][idx],
            "idx" : idx
        }


class IshidaSuiClassifierTrain(IshidaSuiClassifierDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "train"].reset_index(drop=True)


class IshidaSuiClassifierVal(IshidaSuiClassifierDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "valid"].reset_index(drop=True)

    


        

