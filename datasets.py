from torch.utils.data import DataLoader, Dataset, TensorDataset
import cv2
import numpy as np
import torch

from albumentations import Resize, NoOp, Normalize, Compose

def open_image(img_path, tfm=NoOp()):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = tfm(image=img)['image']
    return img

def open_mask(mask_path, tfm=NoOp()):
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    mask = (mask == 255).astype(np.float32)
    mask = tfm(image=mask, mask=mask)['mask']
    return np.prod(mask, axis=-1)

class SaltDataset(Dataset):
    def __init__(self, img_paths, mask_paths, tfms):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        assert len(mask_paths) == len(mask_paths)

    def __getitem__(self, i):
        img = open_image(self.img_paths[i])
        mask = open_mask(self.mask_paths[i])

        res = self.tfms(image=img, mask=mask)
        return res['image'], res['mask']

class SaltDatasetArrays(Dataset):
    def __init__(self, img_tensor, mask_tensor, tfms):
        self.img = img_tensor
        self.mask = mask_tensor
        self.tfms = tfms
        assert self.img.shape[0] == self.mask.shape[0]

    def __getitem__(self, i):
        img = self.img[i]
        mask = self.mask[i]

        res = self.tfms(image=img, mask=mask)
        return res['image'], res['mask']


def create_arrays(img_paths, mask_paths, size=None):
    tfm = NoOp() if size is None else Resize(size, size)
    tfm = Compose([tfm, Normalize()])
    imgs = np.stack([open_image(i, tfm=tfm) for i in img_paths])
    masks = np.stack([open_mask(i, tfm=tfm) for i in mask_paths])
    return torch.Tensor(imgs), torch.Tensor(masks)

def get_dls(train_img_paths,
            train_mask_paths,
            val_img_paths,
            val_mask_paths, size=128,
            batch_size=1, num_workers=1, pin_memory=False):
    trn_img, trn_mask = create_arrays(train_img_paths, train_mask_paths, size)
    val_img, val_mask = create_arrays(val_img_paths, val_mask_paths, size)
    trn_ds = SaltDatasetArrays(trn_img, trn_mask)
    val_ds = SaltDatasetArrays(val_img, val_mask)

    trn_dl = DataLoader(trn_ds, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return trn_dl, val_dl