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
import opacus
import optuna
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from optuna.distributions import BaseDistribution, IntDistribution, FloatDistribution
from PIL import Image

from ldm.data.util import get_sample_rate
from ldm.privacy.myopacus import MyBatchSplittingSampler
from ldm.privacy.privacy_analysis import compute_noise_multiplier
 
# TODO: Implement Opacus batch memory manager 

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
            "filename": [f"control/samples/{i:05d}.png" for i in range(NEG_SIZE)], 
            "label": [0] * NEG_SIZE
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
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
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
        self.dataset = IshidaSuiClassifierDataset(transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            shuffle=self.shuffle, collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, collate_fn=collate)

    def dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, collate_fn=collate)


### LIGHTNING MODULE ###

class ClassifierModule(pl.LightningModule):
    def __init__(self, 
            lr=0.001, 
            dp=False, 
            epsilon=10, 
            delta=1e-5,
            max_batch_size=64, 
            max_grad_norm=1.0, 
            poisson_sampling=True,
            classic_guidance=False,
            weight_decay=0,
            beta1=0.9,
            beta2=0.999,
            h=0,
            wandb=False
        ):
        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.criterion = torch.nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1(num_classes=2)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.wandb = wandb

        self.classic_guidance = classic_guidance
        if self.classic_guidance:
            in_features = self.model.fc.in_features + 1
        else:
            in_features = self.model.fc.in_features
        if h == 0:
            self.model.fc = torch.nn.Linear(in_features, 2)
        else:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, 2)
            )

        self.dp = {
            "enabled" : True,
            "epsilon" : epsilon,
            "delta" : delta,
            "max_batch_size" : max_batch_size,
            "max_grad_norm" : max_grad_norm,
            "poisson_sampling" : poisson_sampling
        } if dp else {"enabled" : False}


    def forward(self, x):
        if not self.classic_guidance:
            return self.model(x)

        raise NotImplementedError


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
            )

        if self.dp["enabled"]:
            privacy_engine = opacus.PrivacyEngine()
            dataloader = self.trainer.datamodule.train_dataloader()
            sample_rate = get_sample_rate(dataloader)
            noise_multiplier = compute_noise_multiplier(
                self.dp, 
                sample_rate,
                self.trainer.max_epochs
            )
            _, optimizer, _ = privacy_engine.make_private(
                module=self,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=self.dp["max_grad_norm"],
                poisson_sampling=self.dp["poisson_sampling"]
            )

        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        if self.wandb:
            wandb.log({"train_loss": loss})
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = self.criterion(logits, y)
        self.log("val_loss", val_loss, prog_bar=True)
        y_pred = torch.argmax(logits, dim=1)
        self.f1(y_pred, y)
        self.log("val_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss


### HYPERPARAMETER TUNING ###

def zero_heavy_distribution(dist):
    if dist == "int":
        parent = IntDistribution
    elif dist == "log_uniform":
        parent = FloatDistribution
    else:
        raise Exception("Invalid distribution type")

    class ZeroHeavyDistribution(parent):
        """
        With probability p, samples 0. With probability 1 - p, samples from distribution
        given by `dist`, `low`, `high`.

        IMPORTANT: If you get an AssertionError in _get_single_value for 
        /optuna/distributions.py, simply comment out the assert statement in your
        version of the optuna source code. 
        """
        def __init__(self, p, low, high):
            self.p = p
            self.low = low 
            self.high = high
            self.step = 1 if dist == "int" else None
            self.log = dist == "log_uniform"
    
        def single(self):
            if np.random.random() < self.p:
                return 0
            else:
                if dist == "int":
                    return np.random.randint(self.low, self.high + 1)
                elif dist == "log_uniform":
                    return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
                else:
                    raise Exception("Invalid distribution type")

        def _contains(self, x):
            return self.low <= x <= self.high

        def to_external_repr(self, x):
            return x

        def to_internal_repr(self, x):
            return x

        def _asdict(self):
            return {"p" : self.p, "distribution": self.dist, "low": self.low, "high": self.high}

        def __repr__(self):
            return f"ZeroHeavyDistribution(params={str({**self._asdict(), 'dist' : dist})})"

    return ZeroHeavyDistribution


def make_objective_function(args):
    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        epochs = trial.suggest_int("epochs", 10, 100)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        hidden_size = trial._suggest("hidden_size", 
                                    zero_heavy_distribution("int")(0.5, 25, 100)) 
        weight_decay = trial._suggest("weight_decay",
                        zero_heavy_distribution("log_uniform")(0.25, 1e-6, 1e-2))

        data_module = IshidaSuiDataModule(
            batch_size=batch_size, 
            shuffle=args.shuffle,
        )
        model = ClassifierModule(
            lr=lr, 
            dp=args.dp_enabled,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            max_batch_size=args.dp_max_batch_size,
            max_grad_norm=args.dp_max_grad_norm,
            poisson_sampling=args.dp_poisson_sampling,
            weight_decay=weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            h=hidden_size,
            wandb=args.wandb
        )

        trainer = Trainer(
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
            max_epochs=args.epochs, 
            gpus=args.gpus)  
        trainer.fit(model, datamodule=data_module)

        return trainer.callback_metrics["val_loss"].item()
    
    return objective


def make_logging_function(args):
    def log_trial(study, trial):
        with open(args.log_file, "a") as f:
            f.write(str({
                "trial_number": trial.number,
                "params": trial.params,
                "val_loss": trial.value
            }) + "\n")

        if args.wandb:
            wandb.log({
                "trial_number": trial.number,
                "hidden_size": trial.params["hidden_size"],
                "lr": trial.params["lr"],
                "weight_decay": trial.params["weight_decay"],
                "batch_size": trial.params["batch_size"],
                "epochs": trial.params["epochs"],
                "val_loss": trial.value
            })

    return log_trial


### MAIN ###

def main(args):
    if args.wandb:
        wandb.init(
            project=args.name,
            config=vars(args)
        )

    if args.train:
        torch.manual_seed(args.seed)

        data_module = IshidaSuiDataModule(
            batch_size=args.batch_size, 
            shuffle=args.shuffle,
        )
        model = ClassifierModule(
            lr=args.lr, 
            dp=args.dp_enabled,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            max_batch_size=args.dp_max_batch_size,
            max_grad_norm=args.dp_max_grad_norm,
            poisson_sampling=args.dp_poisson_sampling,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            h=args.mlp_hidden_size,
            wandb=args.wandb
        )

        checkpoint_callback_top_k = ModelCheckpoint(
            monitor="val_f1",               
            dirpath=f"outputs/sotonami/{args.name}/",           
            filename="checkpoint-{epoch:02d}",
            save_top_k=3,                     
            mode="max",                       
            save_weights_only=True,           
            verbose=True)
        checkpoint_callback_last = ModelCheckpoint(
            dirpath=f"outputs/sotonami/{args.name}/",
            filename="last",
            save_top_k=1,                     
            save_last=True,                   
            save_weights_only=True,           
            verbose=True)
        early_stopping_callback = EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=args.patience,
            strict=True,
            verbose=True)
        
        trainer = Trainer(
            callbacks=[checkpoint_callback_top_k, checkpoint_callback_last,
                        early_stopping_callback],
            max_epochs=args.epochs, 
            gpus=args.gpus)  
        trainer.fit(model, datamodule=data_module)

        wandb.finish()

    elif args.test:
        raise NotImplementedError

    elif args.tune:
        objective = make_objective_function(args)
        log_trial = make_logging_function(args)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, callbacks=[log_trial])
        print(f"Best trial: {study.best_trial.value}")
        print("Best hyperparameters: ", study.best_trial.params)
       

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true",
                        help="whether to train a model")
    parser.add_argument("--tune", action="store_true",
                        help="whether to perform hyperparameter tuning")
    parser.add_argument("--test", action="store_true",
                        help="whether to run the model")
    parser.add_argument("--wandb", action="store_true",
                        help="whether to use wandb (still WIP)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of gpus")
    parser.add_argument("--epochs", type=int, default=10,
                        help="max number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--name", type=str, default=timestamp,
                        help="name of current run for output directory + wandb")
    parser.add_argument("--patience", type=int, default=5,
                        help="see PyTorch")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--shuffle", action="store_true",
                        help="whether to shuffle the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for random operations")
    parser.add_argument("--classic_guidance", action="store_true",
                        help="whether to add noise and noise level as input of traditional classifier guidance")
    parser.add_argument("--dp_enabled", action="store_true",
                        help="whether to use dp")
    parser.add_argument("--dp_epsilon", type=float, default=10)
    parser.add_argument("--dp_delta", type=float, default=1.0e-05)
    parser.add_argument("--dp_max_batch_size", type=int, default=64,
                        help="maximum physical batch size when using dp")
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dp_poisson_sampling", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 in Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 in Adam optimizer")
    parser.add_argument("--mlp_hidden_size", type=int, default=0,
                        help="if 0 then no hidden layer else hidden layer size")
    parser.add_argument("--log_file", type=str, default="output.txt",
                        help="logging file for hyperparameter tuning")
    
    args = parser.parse_args()
    main(args)