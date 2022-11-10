from scripts.dataset.GRL.dataset_GRL import CulturalSiteDatasetGRL
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
# Global Variable
BASE_CLASS_DATASETS_PATH = os.path.join(os.getcwd(), 'CLASS-EGO-CH-OBJ-ADAPT')

class CulturalSiteDataModuleGRL(pl.LightningDataModule):

    FIT_STAGE = 0
    TEST_STAGE = 1
    ALL_STAGE = 2

    def __init__(self, batch_size, num_workers= 1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cultural_site_train = None
        self.cultural_site_test = None            

    def setup(self, stage=ALL_STAGE):
        if stage==CulturalSiteDataModuleGRL.FIT_STAGE or stage==CulturalSiteDataModuleGRL.ALL_STAGE:
            self.cultural_site_train = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.TRAIN)

            train_set = self.cultural_site_train.get_real_image_dataset()
            new_train_set, new_validation_set = train_test_split(train_set, test_size = 0.1) # splitting train ratio 90/10
            self.cultural_site_train.set_real_image_dataset(new_train_set)
            
            self.cultural_site_val = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.VALIDATION)
            self.cultural_site_val.set_real_image_dataset(new_validation_set)

        if stage==CulturalSiteDataModuleGRL.TEST_STAGE or stage==CulturalSiteDataModuleGRL.ALL_STAGE:
            self.cultural_site_test = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.TEST)

    def filter_train(self, pixel_threshold):
        self.cultural_site_train.filter_dataset(pixel_threshold)

    def filter_val(self, pixel_threshold):
        self.cultural_site_val.filter_dataset(pixel_threshold)

    def filter_test(self, pixel_threshold):
        self.cultural_site_test.filter_dataset(pixel_threshold)
    
    def train_dataloader(self):
        return DataLoader(self.cultural_site_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.cultural_site_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cultural_site_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def set_train_trasform(self, transform):
        if(self.cultural_site_train is not None):
            self.cultural_site_train.set_transform(transform)
    
    def set_val_trasform(self, transform):
         if(self.cultural_site_val is not None):
            self.cultural_site_val.set_transform(transform)
        
    def set_test_trasform(self, transform):
        if(self.cultural_site_test is not None):
            self.cultural_site_test.set_transform(transform)

    def calculate_train_mean_and_std(self, resize_min_size):
        img_dataset = np.array([self.cultural_site_train.get_syn_image_dataset()[:,1], self.cultural_site_train.get_syn_image_dataset()[:,1]])
        means = []
        for img_path in img_dataset:
            resized_img = self.__resize_img(img_path, resize_min_size)
            image_rgb = np.array(resized_img) / 255
            means.append(np.mean(image_rgb, axis=(0,1)))
        mean_rgb = np.mean(means, axis=0)  # mu_rgb.shape == (3,)

        variances = []
        for img_path in img_dataset:
            resized_img = self.__resize_img(img_path, resize_min_size)
            image_rgb = np.array(resized_img) / 255
            var = np.mean((image_rgb - mean_rgb) ** 2, axis=(0,1))
            variances.append(var)
        std_rgb = np.sqrt(np.mean(variances, axis=0))  # std_rgb.shape == (3,)

        return mean_rgb, std_rgb

    def __resize_img(self, img_path, min_size):
        img = Image.open(img_path)
        aspect = img.width / img.height if img.width > img.height else img.height / img.width               
        return np.asarray(img.resize((int(min_size * aspect), min_size), Image.BICUBIC) if img.width > img.height else img.resize((min_size, int(min_size * aspect)), Image.BICUBIC))