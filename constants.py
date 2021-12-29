import os
import torch

BASE_DIR = os.path.abspath(os.getcwd())

MODEL_DIR = os.path.join(BASE_DIR, 'model')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'train_masks')
TEST_DIR = os.path.join(BASE_DIR, 'test')

DEVICE = torch.device('cuda')
