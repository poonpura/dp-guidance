import io
import random
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image 

def preprocess_image(png_bytes, interpolation, transforms):
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    assert image.size[0] == image.size[1]
    size, _ = image.size
    image_resized = image.resize((256, 256), interpolation)
    tensor = transforms(image_resized)
    return tensor

def preprocess_string(s):
    if "Pokemon" in s:
        return s
    nouns = ["animal", "cartoon character", "creature", "monster", "character"]
    for noun in nouns:
        s = s.replace("noun", "Pokemon")
    if "Pokemon" in s:
        return s
    s = s.replace("flower with", "flower Pokemon with")
    if "Pokemon" in s:
        return s
    species = ["bird", "dog", "dragon", "sheep", "dinosaur", "butterfly", 
                "turtle", "bee", "shark", "whale", "pig", "rabbit", "insect", 
                "toy", "snake", "sun", "moon", "crab", "lizard", "ghost", "seal",
                "robot", "cat", "serpent", "monkey", "lion", "bat", "eggplant",
                "bunny", "alligator", "frog", "spider", "car", "mouse", 
                "ice cream cone", "deer", "alien", "chandelier", "bear", "mermaid",
                "plant", "horse"]
    species = list(dict.fromkeys(species)) # remove duplicates
    for x in species:
        s = s.replace(x, f"{x} Pokemon")
        if "Pokemon" in s:
            return s 
    return s

class PokemonDataset(Dataset):
    def __init__(self, size=None, interpolation="bicubic", split=0.9, seed=25, **kwargs):
        df = pd.read_parquet("/lfs/skampere1/0/pura/datasets/pokemon-llava-captions/data/train-00000-of-00001-dd72dfb2bf009aab.parquet")
        self.interpolation = {"linear": Image.BILINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.data = pd.DataFrame()
        self.data["image"] = df["image"].map(lambda x: x["bytes"])
        self.data["text"] = df["text"]

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
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()
        ])
        self.val_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data["split"][idx] == "train":
            transforms = self.train_transforms
        else:
            transforms = self.val_transforms
        return {
            "image" : preprocess_image(self.data["image"][idx], 
                                                self.interpolation, transforms),
            "caption" : preprocess_string(self.data["text"][idx])
        }


class PokemonTrain(PokemonDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "train"].reset_index(drop=True)


class PokemonVal(PokemonDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "valid"].reset_index(drop=True)

    


        

