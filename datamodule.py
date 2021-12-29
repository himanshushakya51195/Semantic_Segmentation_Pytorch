from abc import ABC
from typing import Optional, Union, List
import torch
import pytorch_lightning as pl
import math
from torch.utils.data import DataLoader


class SegDataModule(pl.LightningDataModule, ABC):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 val_fraction: float,
                 test_fraction: float,
                 n_workers: int = 0):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.n_workers = n_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        num_sample = len(self.dataset)
        num_test_sample = math.ceil(num_sample * self.test_fraction)
        num_val_sample = math.ceil(num_sample * self.val_fraction)
        num_train_sample = num_sample - num_val_sample - num_test_sample

        train, val, test = torch.utils.data.random_split(
            self.dataset,
            [num_train_sample, num_val_sample, num_test_sample]
        )

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def train_dataloader(self, *args, **kwagrs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          self.batch_size,
                          num_workers=self.n_workers,
                          shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset,
                          self.batch_size,
                          num_workers=self.n_workers,
                          shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          self.batch_size,
                          num_workers=self.n_workers,
                          shuffle=False)
