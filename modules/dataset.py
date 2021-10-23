import os
import numpy as np
import cv2
from torch.utils.data import Dataset


class PlantDataset(Dataset):
    '''
    meta : train/test.csv
    mode : train/valid/test
    '''

    def __init__(self, cfg, meta, transforms=None, augmentations=None, mode='train'):
        super().__init__()
        self.cfg = cfg
        self.meta = meta
        self.mode = mode
        self.transforms = transforms
        self.augmentations = augmentations

    def __getitem__(self, index: int):
        img_info = self.meta.iloc[index, :]
        img = cv2.imread(os.path.join(
            self.cfg['PATH']['DATA'], img_info['img_path']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Histogram Equalization
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb_planes = cv2.split(img_ycrcb)
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        dst_ycrcb = cv2.merge(ycrcb_planes)
        dst_img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2RGB)
        dst_img = (dst_img/255.).astype('float32')

        label = img_info['disease_code'].astype(np.uint8)
        sample = {'image': dst_img, 'label': label}
        if self.mode == 'train' and self.augmentations:
            sample['image'] = self.augmentations(
                image=sample['image'])['image']

        if self.transforms:
            sample['image'] = self.transforms(image=sample['image'])

        return sample

    def __len__(self):
        return len(self.meta)


class PlantTestDataset(Dataset):
    '''
    meta : train/test.csv
    mode : train/valid/test
    '''

    def __init__(self, cfg, meta, transforms=None):
        super().__init__()
        self.cfg = cfg
        self.meta = meta
        self.transforms = transforms

    def __getitem__(self, index: int):
        img_info = self.meta.iloc[index, :]
        img = cv2.imread(os.path.join(
            self.cfg['PATH']['DATA'], img_info['img_path']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Histogram Equalization
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb_planes = cv2.split(img_ycrcb)
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        dst_ycrcb = cv2.merge(ycrcb_planes)
        dst_img = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2RGB)
        dst_img = (dst_img/255.).astype('float32')

        sample = {'image': dst_img}

        if self.transforms:
            sample['image'] = self.transforms(image=sample['image'])

        return sample

    def __len__(self):
        return len(self.meta)
