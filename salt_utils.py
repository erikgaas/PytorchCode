import random
from albumentations import Resize, Normalize, RandomRotate90, Flip, Compose
from sklearn.model_selection import KFold
import numpy as np

random.seed(42)

def get_paths(dataset_path):
    train_names = [i.name for i in (dataset_path / 'train' / 'images').iterdir()]
    train_images = [str(dataset_path / 'train' / 'images' / i) for i in train_names]
    train_masks = [str(dataset_path / 'train' / 'masks' / i) for i in train_names]
    return np.array(train_images), np.array(train_masks)


def get_tfms(size, val=False):
    augs = [Resize(size, size), Normalize()]
    if val is False:
        augs += [RandomRotate90(), Flip()]
    return Compose(augs)

def cross_val_paths(dataset_path):
    img_paths, mask_paths = get_paths(dataset_path)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    for trn, val in kfold.split(img_paths, mask_paths):
        yield img_paths[trn], mask_paths[trn], img_paths[val], mask_paths[val]

