import io
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image 

### DATASET ###

PATH = "/lfs/skampere1/0/pura/datasets/sotonami/"
NEG_SIZE = 1640

def preprocess_image(filename, interpolation, transforms):
    image = Image.open(filename).convert("RGB")
    assert image.size[0] == image.size[1]
    size, _ = image.size
    image = image.resize((256, 256), interpolation)
    image = transforms(image)
    return image

def index_map(i):
    raw_index = i // 10
    if raw_index <= 111:
        return raw_index
    elif raw_index <= 117:
        return raw_index + 2
    else:
        return raw_index + 3


class IshidaSuiClassifierDataset(Dataset):
    def __init__(self, size=None, interpolation="bicubic", split=0.9, seed=25, **kwargs):
        self.interpolation = {"linear": Image.BILINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]

        # loading positives 
        df = pd.read_csv(PATH + "data/metadata.csv")
        pos = pd.DataFrame()
        pos["filename"] = df["file_name"].map(lambda x: f"data/{x}")
        pos["label"] = 1

        # loading negatives
        neg = pd.DataFrame({
            'filename': [f"control/samples/{i:05d}.png" for i in range(NEG_SIZE)], 
            'label': [0] * NEG_SIZE
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data["split"][idx] == "train":
            transforms = self.train_transforms
        else:
            transforms = self.val_transforms
        path = PATH + self.data["filename"][idx]
        return {
            "image" : preprocess_image(path, self.interpolation, transforms),
            "label" : self.data["label"][idx]
        }


class IshidaSuiClassifierTrain(IshidaSuiClassifierDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "train"].reset_index(drop=True)


class IshidaSuiClassifierVal(IshidaSuiClassifierDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.data[self.data["split"] == "valid"].reset_index(drop=True)


### DATALOADER ###

def collate(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return images, labels


class IshidaSuiDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, shuffle):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = IshidaSuiClassifierTrain(transform=self.transform)
        self.val_dataset = IshidaSuiClassifierVal(transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            shuffle=self.shuffle, collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, collate_fn=collate)


### LIGHTNING MODULE ###

class ClassifierModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2) 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        val_loss = self.criterion(logits, y)
        self.log('val_loss', val_loss)

        y_pred = torch.argmax(logits, dim=1)
        f1 = self.f1_score(y_pred, y)
        self.log('val_f1', f1)

        return val_loss


### MAIN ###

def main(args):
    if args.train:
        torch.manual_seed(args.seed)

        data_module = IshidaSuiDataModule(batch_size=args.batch_size, 
                                            shuffle=args.shuffle)
        model = ClassifierModule()

        checkpoint_callback_top_k = ModelCheckpoint(
            monitor='val_f1',               
            dirpath=f'outputs/sotonami/{args.name}/',           
            filename='checkpoint-{epoch:02d}',
            save_top_k=3,                     
            mode='max',                       
            save_weights_only=True,           
            verbose=True)
        checkpoint_callback_last = ModelCheckpoint(
            dirpath=f'outputs/sotonami/{args.name}/',
            filename='last',
            save_top_k=1,                     
            save_last=True,                   
            save_weights_only=True,           
            verbose=True)
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            strict=True,
            verbose=True)
        
        trainer = Trainer(
            callbacks=[checkpoint_callback_top_k, checkpoint_callback_last,
                        early_stopping_callback],
            max_epochs=args.epochs, 
            gpus=args.gpus)  
        trainer.fit(model, datamodule=data_module)

    elif args.test:
        raise NotImplementedError
       

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true",
                        help="whether to train a model")
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of gpus")
    parser.add_argument("--epochs", type=int, default=10,
                        help="max number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--name", type=str, default=timestamp,
                        help="name of current run for output directory")
    parser.add_argument("--test", action="store_true",
                        help="whether to run the model")
    parser.add_argument("--patience", type=int, default=5,
                        help="see PyTorch")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--shuffle", action="store_true",
                        help="whether to shuffle the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for random operations")
    
    args = parser.parse_args()
    main(args)