import abc
import datetime as dt
import math
import pathlib
from typing import TypeVar, Type
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.utils.data
import yaml
import datamodule

T = TypeVar('T')


class SegTrainingModule(pl.LightningModule, metaclass=abc.ABCMeta):
    underlying_model: torch.nn.Module
    lr: float
    lr_decay_rate: float = None
    lr_step_size: int = None
    init_state: torch.nn.Module = None

    def save_model(self, save_dir: pathlib.Path):
        save_dir.mkdir(exist_ok=True)
        torch.save(self.underlying_model, save_dir / 'model.pth')
        with open(save_dir / 'hparams.yaml', 'w+') as fp:
            yaml.dump(dict(self.hparams), fp)

    @classmethod
    @abc.abstractmethod
    def from_scratch(cls: Type[T], lr: float, *args, **kwargs) -> T:
        pass

    @classmethod
    @abc.abstractmethod
    def transfer_learning(cls: Type[T],
                          lr: float,
                          lr_decay_rate: float,
                          init_state_path: pathlib.Path,
                          init_hparams_path: pathlib.Path) -> T:
        pass


class TrainModuleRunner(object):
    def __init__(self,
                 training_module: SegTrainingModule,
                 n_epochs: int,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 n_workers: int,
                 val_fraction: float,
                 test_fraction: float,
                 model_name: str,
                 model_root_dir: pathlib.Path):
        self.model_name = model_name
        self.model_dir = pathlib.Path(model_root_dir) / self.model_name
        self.training_module = training_module
        self.datamodule = datamodule.SegDataModule(dataset=dataset,
                                                   batch_size=batch_size,
                                                   n_workers=n_workers,
                                                   val_fraction=val_fraction,
                                                   test_fraction=test_fraction)

        self.checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=5)
        early_stopper = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0.2,
                                                   patience=3,
                                                   stopping_threshold=0.08)
        tb_logger = pl_loggers.TensorBoardLogger(str(self.model_dir / 'logs'))
        self.trainer = pl.Trainer(
            min_epochs=n_epochs,
            max_epochs=math.ceil(1.5 * n_epochs),
            gpus=-1 if torch.cuda.is_available() else None,
            accelerator='dp' if torch.cuda.is_available() else None,
            auto_select_gpus=True,
            default_root_dir=str(self.model_dir / 'checkpoints'),
            logger=tb_logger,
            profiler='simple',
            callbacks=[self.checkpointer, early_stopper]
        )

    def run(self):
        self.trainer.fit(self.training_module, datamodule=self.datamodule)
        self.training_module = self.training_module.load_from_checkpoint(self.checkpointer.best_model_path)
        self.trainer.test(self.training_module, datamodule=self.datamodule, ckpt_path=None)
        torch.save({
            'name': self.model_name,
            'save_time_utc': dt.datetime.utcnow(),
            'model': self.training_module.underlying_model,
            'hparams': self.training_module.hparams,
            'preprocessor': self.datamodule.dataset.preprocessor,
            'epochs': self.training_module.current_epoch,
            'lr': self.training_module.lr,
            'lr_step_size': self.training_module.lr_step_size,
            'lr_decay_rate': self.training_module.lr_decay_rate,
            'dataset_size': len(self.datamodule.dataset)
        }, self.model_dir / 'model.pth')
