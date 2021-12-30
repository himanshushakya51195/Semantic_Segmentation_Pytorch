import itertools
import pathlib
import torch.utils.data
from typing import Tuple
from PIL import Image
import numpy as np
import torchvision.transforms as T


class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_dir: pathlib.Path,
                 train_folder: str = 'train',
                 mask_folder: str = 'train_masks',
                 nn_img_shape: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.root_dir = pathlib.Path(root_dir)
        self.train_folder = self.root_dir / train_folder
        self.mask_folder = self.root_dir / mask_folder
        self.nn_img_shape = nn_img_shape

        jpg_image_path = self.train_folder.glob('*.jpg')
        png_image_path = self.train_folder.glob('*.png')

        image_path = itertools.chain(jpg_image_path, png_image_path)

        self.image_path = sorted(image_path)

        self.mask_path = sorted(self.mask_folder.glob('*mask.gif'))

        assert len(self.image_path) == len(self.mask_path), (
            f'expected {len(self.image_path)} images and {len(self.image_path)} masks,'
            f'but found {len(self.mask_path)} masks')

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        input_img_path = self.image_path[item]
        mask_path = self.mask_path[item]

        assert input_img_path.stem + '_mask' == mask_path.stem, (
            f'expected file name {input_img_path.stem}_mask{mask_path.suffix}'
            f' but found {mask_path.name}')

        image = Image.open(str(input_img_path))
        mask = Image.open(str(mask_path))

        image = image.resize(self.nn_img_shape, Image.NEAREST)
        mask = mask.resize(self.nn_img_shape, Image.NEAREST)
        mask = mask.convert("L")

        image = np.array(image)
        mask = np.array(mask)

        image2tensor = T.ToTensor()
        image = image2tensor(image)
        mask = image2tensor(mask)
        mask = torch.reshape(mask, (512, 512))
        mask = mask.int()
        return image, mask
