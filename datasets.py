from torch.utils.data import DataLoader, Dataset
import cv2

class SaltDataset(Dataset):
    def __init__(self, img_paths, mask_paths, tfms):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        assert len(mask_paths) == len(mask_paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.img_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        res = self.tfms(image=img, mask=mask)
        return res['image'], res['mask']