import pathlib
from abc import ABC
from trainmodule import SegTrainingModule

import yaml
import torch
import segmentation_models_pytorch as smp
import metrics as M


class SegTrainingModule2(SegTrainingModule, ABC):
    def __init__(self,
                 arch: str,
                 classes: int,
                 in_channels: int,
                 encoder_name: str,
                 encoder_weights: str,
                 encoder_depth: int = 5,
                 lr: float = 0.0001,
                 lr_delay_rate: float = None,
                 lr_step_size: int = None,
                 init_state_path: pathlib.Path = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_model = smp.create_model(arch=arch,
                                                 classes=classes,
                                                 in_channels=in_channels,
                                                 encoder_name=encoder_name,
                                                 encoder_weights=encoder_weights,
                                                 encoder_depth=encoder_depth,
                                                 activation='softmax2d')

        self.transfer_mode = False
        if init_state_path is not None:
            assert lr_delay_rate is not None, f'{lr_delay_rate.__name__} is required if ' f'{init_state_path.__name__} is given '
            assert lr_step_size is not None, f'{lr_delay_rate.__name__} is required if ' f'{init_state_path.__name__} is given'

            self.transfer_mode = True
            self.lr_decay_rate = lr_delay_rate
            self.lr_step_size = lr_step_size
            init_state = torch.load(init_state_path, map_location=self.device)
            self.underlying_model.load_state_dict(init_state.state_dict())

        self.lr = lr
        self.smp_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=False)
        self.smp_metrics = [M.IOU(activation='argmax2d')]

        self.save_hyperparameters('arch',
                                  'classes',
                                  'in_channels',
                                  'encoder_name',
                                  'encoder_weights',
                                  'encoder_depth')

    @classmethod
    def from_scratch(cls,
                     lr: float,
                     model_arch: str,
                     classes: int,
                     in_channels: int,
                     encoder_name: str,
                     encoder_weights: str,
                     encoder_depth: int,
                     *args, **kwargs):

        lit_model = cls(arch=model_arch,
                        classes=classes,
                        in_channels=in_channels,
                        encoder_name=encoder_name,
                        encoder_weights=encoder_weights,
                        encoder_depth=encoder_depth,
                        lr=lr)
        return lit_model

    @classmethod
    def transfer_learning(cls,
                          lr: float,
                          lr_decay_rate: float,
                          lr_step_size: int,
                          init_state_path: pathlib.Path,
                          init_hparams_path: pathlib.Path):
        with open(init_hparams_path, 'r') as fp:
            hparams = yaml.full_load(fp)
        lit_model = cls(**hparams,
                        lr=lr,
                        lr_decay_rate=lr_decay_rate,
                        lr_step_size=lr_step_size,
                        init_state_path=init_state_path)
        return lit_model

    def forward(self, x):
        y_hat = self.underlying_model(x)
        return y_hat

    def training_step(self, batch, batch_id):
        x, y_true = batch
        output = self.underlying_model(x)
        loss = self.smp_loss(output, y_true.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        x, y_true = batch
        output = self.underlying_model(x)
        loss = self.smp_loss(output, y_true.long())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.smp_metrics:
            score = metric(output, y_true)
            self.log(f'val_{metric.__name__}', score,
                     on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_id):
        x, y_true = batch
        output = self.underlying_model(x)
        loss = self.smp_loss(output, y_true.long())
        # y_hat = output.argmax(dim=1).float()
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.smp_metrics:
            # score = iou(y_hat, y_true)
            score = metric(output, y_true)
            self.log(f'test_{metric.__name__}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.transfer_mode:
            # create a step lr if transfer learning with a pretrained model
            optim = torch.optim.Adam(params=self.underlying_model.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.StepLR(optim, step_size=self.lr_step_size, gamma=self.lr_decay_rate)
            return [optim], [sched]
        else:
            # create a simple optimizer if training a greenfield model
            optim = torch.optim.Adam(params=self.underlying_model.parameters(), lr=self.lr)
            return optim
