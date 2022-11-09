import random
from PIL import Image
import os
import glob
from typing import Callable, Optional
from torch.utils.data import Dataset

class CulturalSiteDatasetGAN(Dataset):

    TRAIN = "training"
    TEST = "test"

    REAL = "real"
    SYNTHETIC = "syntehtic"

    def __init__(self, dataset_base_path, transform: Optional[Callable] = None, mode= TRAIN) -> None:
        self.transform = transform

        # ottieni i path delle immagini in A e B
        self.images_path_syn = sorted(glob.glob(os.path.join(dataset_base_path, CulturalSiteDatasetGAN.SYNTHETIC, '%s' % mode, 'data') + '/*.*'))
        self.images_path_real = sorted(glob.glob(os.path.join(dataset_base_path, CulturalSiteDatasetGAN.REAL, '%s' % mode, 'data') + '/*.*'))

    def __check_filter_size(self, img_path, pixel_threshold):
        w, h = Image.open(img_path).size
        return w > pixel_threshold and h > pixel_threshold

    def filter_dataset(self, pixel_threshold):
        tmp = [x for x in self.images_path_syn if self.__check_filter_size(x, pixel_threshold)]
        self.images_path_syn = tmp
        tmp = [x for x in self.images_path_real if self.__check_filter_size(x, pixel_threshold)]
        self.images_path_real = tmp

    def __getitem__(self, index: int):
        #apro l'iesima immagine A (uso il modulo per evitare di sforare)
        item_A = Image.open(self.images_path_syn[index % len(self.images_path_syn)])
        #apro una immagine B a caso
        item_B = Image.open(self.images_path_real[random.randint(0, len(self.images_path_real) - 1)])
        
        if self.transform is not None:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return item_A, item_B

    def __len__(self) -> int:
        return max(len(self.images_path_syn), len(self.images_path_real))