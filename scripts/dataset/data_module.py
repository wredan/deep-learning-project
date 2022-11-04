import json
from scripts.dataset.dataset import CulturalSiteDataset
from scripts.dataset.dataset_loader import CulturalSiteDatasetsLoader
import os
from matplotlib import pyplot as plt
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

# Global Variable
DOWNLOAD_URL = "https://iplab.dmi.unict.it/EGO-CH-OBJ-ADAPT/EGO-CH-OBJ-ADAPT.zip"
BASE_CLASS_DATASETS_PATH = os.path.join(os.getcwd(), 'CLASS-EGO-CH-OBJ-ADAPT')
BASE_MAIN_DATASETS_PATH = os.path.join(os.getcwd(), 'EGO-CH-OBJ-ADAPT')
ZIP_FILE_PATH = os.path.join(os.getcwd(), 'EGO-CH-OBJ-ADAPT.zip')

class CulturalSiteDataModule(pl.LightningDataModule):

    FIT_STAGE = 0
    TEST_STAGE = 1
    ALL_STAGE = 2

    REAL_DATASET = 0
    SYNTHETIC_DATASET = 1

    def __init__(self, batch_size, dataset_type, num_classes, num_workers= 1):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.cultural_site_train = None
        self.cultural_site_val = None
        self.cultural_site_test = None
        self.load_class_names()
                
    def load_class_names(self):
        class_file = open(os.path.join(os.getcwd(), "utils", "image_classes.json"))
        json_class = json.load(class_file)
        self.class_names = []
        for el in json_class["categories"]:
            self.class_names.append(el["name"])
    
    def prepare_data(self):
        CulturalSiteDatasetsLoader(
            download_path= DOWNLOAD_URL,
            main_dataset_path= BASE_MAIN_DATASETS_PATH,
            class_dataset_path= BASE_CLASS_DATASETS_PATH,
            zip_file_path= ZIP_FILE_PATH)

    def setup(self, stage=ALL_STAGE):
        # Assign train/val datasets for use in dataloaders
        if stage == CulturalSiteDataModule.FIT_STAGE or stage == CulturalSiteDataModule.ALL_STAGE:
            self.cultural_site_train = CulturalSiteDataset(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDataset.TRAIN, 
                dataset_type=self.dataset_type)
            if self.dataset_type == CulturalSiteDataModule.SYNTHETIC_DATASET:
                self.cultural_site_val = CulturalSiteDataset(
                    dataset_base_path=BASE_CLASS_DATASETS_PATH,
                    dataset_stage=CulturalSiteDataset.VALIDATION, #TODO: che succede se è nullo?
                    dataset_type=self.dataset_type)
            else:
                self.cultural_site_val = CulturalSiteDataset(
                    dataset_base_path=BASE_CLASS_DATASETS_PATH,
                    dataset_stage=CulturalSiteDataset.TEST, 
                    dataset_type=self.dataset_type)

        # Assign test dataset for use in dataloader(s)
        if stage == CulturalSiteDataModule.TEST_STAGE or stage == CulturalSiteDataModule.ALL_STAGE:
            self.cultural_site_test = CulturalSiteDataset(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDataset.TEST, 
                dataset_type=self.dataset_type)
    
    def train_preanalysis(self, subplot, title):
        return self.__ds_preanalysis(self.cultural_site_train, subplot, title)

    def val_preanalysis(self, subplot, title):
        return self.__ds_preanalysis(self.cultural_site_val, subplot, title)

    def test_preanalysis(self, subplot, title):
        return self.__ds_preanalysis(self.cultural_site_test, subplot, title)
    
    def get_train_dataset(self):
        return self.cultural_site_train.get_image_dataset()

    def get_val_dataset(self):
        return self.cultural_site_val.get_image_dataset()

    def get_test_dataset(self):
        return self.cultural_site_test.get_image_dataset()

    def __ds_preanalysis(self, dataset: CulturalSiteDataset, subplot, title):
        image_dataset = dataset.get_image_dataset()
        _, counts = np.unique(image_dataset[:, 2], return_counts=True)
        df = pd.DataFrame({
            "Name": self.class_names,
            "Value": counts
        })

        text_limit = 25

        df["Name"] = [label_text[:text_limit] + (label_text[text_limit:] and '..') for label_text in df["Name"]]


        df_sorted = df.sort_values('Value', ascending=False)
        subplot.tick_params(axis='x', labelrotation=45)
        for tick in subplot.get_xticklabels():
            tick.set_horizontalalignment('right')
        subplot.bar('Name', 'Value', data=df_sorted)
        subplot.grid()
        subplot.set_title(title)
        return subplot

    def filter_train(self, soglia_pixel):
        self.cultural_site_train.filter_dataset(soglia_pixel)

    def filter_val(self, soglia_pixel):
        self.cultural_site_val.filter_dataset(soglia_pixel)

    def filter_test(self, soglia_pixel):
        self.cultural_site_test.filter_dataset(soglia_pixel)

    def resize_train(self, min_size):
        prev_image = self.cultural_site_train.get_image_dataset()[0][1]
        self.cultural_site_train.resize_dataset(min_size)
        post_image = self.cultural_site_train.get_image_dataset()[0][1]
        _, (prev_subplot, post_subplot) = plt.subplots(1, 2)
        prev_subplot.set_title("Immagine originale " + str(prev_image.shape[0]) + "x" + str(prev_image.shape[1]) + ":")
        post_subplot.set_title("Immagine dopo resize " + str(post_image.shape[0]) + "x" + str(post_image.shape[1]) + ":")
        prev_subplot.imshow(prev_image)
        post_subplot.imshow(post_image)

    def resize_val(self, min_size):
        self.cultural_site_val.resize_dataset(min_size)

    def resize_test(self, min_size):
        self.cultural_site_test.resize_dataset(min_size)

    def set_train_trasform(self, transform):
        if(self.cultural_site_train is not None):
            self.cultural_site_train.set_transform(transform)
    
    def set_val_trasform(self, transform):
         if(self.cultural_site_val is not None):
            self.cultural_site_val.set_transform(transform)
        
    def set_test_trasform(self, transform):
        if(self.cultural_site_test is not None):
            self.cultural_site_test.set_transform(transform)
    
    def train_dataloader(self):
        return DataLoader(self.cultural_site_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cultural_site_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cultural_site_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def calculate_train_mean_and_std(self, resize_min_size):
        img_dataset = np.array(self.cultural_site_train.get_image_dataset()[:,1])

        # images_rgb = [np.array(img) / 255. for img in img_dataset]
        # # Each image_rgb is of shape (n, 3), 
        # # where n is the number of pixels in each image,
        # # and 3 are the channels: R, G, B.

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