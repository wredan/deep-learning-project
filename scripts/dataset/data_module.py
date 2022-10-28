from sys import stderr
from scripts.dataset.dataset import CulturalSiteDataset
from scripts.dataset.dataset_loader import CulturalSiteDatasetsLoader
import os
from matplotlib import pyplot as plt
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

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
    
    def prepare_data(self):
        CulturalSiteDatasetsLoader(
            download_path= DOWNLOAD_URL,
            main_dataset_path= BASE_MAIN_DATASETS_PATH,
            class_dataset_path= BASE_CLASS_DATASETS_PATH,
            zip_file_path= ZIP_FILE_PATH)

    def setup(self, stage=ALL_STAGE):
        # TODO: qui istanziare CulturalSiteDataset, creare i dataset da passare ai dataloader sotto
        # Assign train/val datasets for use in dataloaders
        if stage == CulturalSiteDataModule.FIT_STAGE or stage == CulturalSiteDataModule.ALL_STAGE:
            self.cultural_site_train = CulturalSiteDataset(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDataset.TRAIN, 
                dataset_type=self.dataset_type)
            if self.dataset_type == CulturalSiteDataModule.SYNTHETIC_DATASET:
                self.cultural_site_val = CulturalSiteDataset(
                    dataset_base_path=BASE_CLASS_DATASETS_PATH,
                    dataset_stage=CulturalSiteDataset.VALIDATION, 
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
    
    def get_dataset(self):
        return self.cultural_site_train.get_image_dataset()

    def __ds_preanalysis(self, dataset: CulturalSiteDataset, subplot, title):
        image_dataset = dataset.get_image_dataset()

        unique, counts = np.unique(image_dataset[:, 2], return_counts=True)

        temp_plot_data = np.column_stack((unique,counts))
        x_sorted_desc = temp_plot_data[temp_plot_data[:, 1].argsort()[::-1]]
        temp_plot_data = np.column_stack((unique,x_sorted_desc))
        subplot.plot(temp_plot_data[:, 0], temp_plot_data[:, 2])
        subplot.get_xaxis().set_visible(False)

        for index, value in enumerate(list(temp_plot_data)):
            subplot.text(index + 0.2, temp_plot_data[-1][2]/2, str(value[1]))
            if index > 0:
                subplot.fill_between(
                    np.array([temp_plot_data[index - 1][0], temp_plot_data[index][0]], dtype=float), 
                    np.array([temp_plot_data[index - 1][2], temp_plot_data[index][2]], dtype=float))
            if index == 15:
                subplot.fill_between(
                    np.array([temp_plot_data[index][0], temp_plot_data[index][0] + 1], dtype=float), 
                    np.array([temp_plot_data[index][2], temp_plot_data[index][2]], dtype=float))

        subplot.set_title(title)
        return subplot
        # print(np.column_stack((unique,counts)))
        # print("len: ", len(image_dataset))

    def filter_train(self, soglia_pixel):
        self.cultural_site_train.filter_dataset(soglia_pixel)

    def filter_val(self, soglia_pixel):
        self.cultural_site_val.filter_dataset(soglia_pixel)

    def filter_test(self, soglia_pixel):
        self.cultural_site_test.filter_dataset(soglia_pixel)

    def resize_train(self):
        prev_image = self.cultural_site_train.get_image_dataset()[0][1]
        self.cultural_site_train.resize_dataset()
        dataset = self.cultural_site_train.get_image_dataset()
        fig, (prev_subplot, post_subplot) = plt.subplots(1, 2)
        prev_subplot.set_title("Immagine originale " + str(prev_image.shape[0]) + "x" + str(prev_image.shape[1]) + ":")
        post_subplot.set_title("Immagine dopo resize " + str(CulturalSiteDataset.RESIZE_HEIGHT) + "x" + str(CulturalSiteDataset.RESIZE_WIDTH) + ":")
        prev_subplot.imshow(prev_image)
        post_subplot.imshow(dataset[0][1])      
        print(dataset[0][0], dataset[0][2])      

    def resize_val(self):
        self.cultural_site_val.resize_dataset()

    def resize_test(self):
        self.cultural_site_test.resize_dataset()

    def normalize_train(self):
        dataset = self.cultural_site_train.get_image_dataset()
        print(dataset[:, 1][:, 0])
        print()
        print(dataset[:, 1][:, 1])
        print()
        print(dataset[:, 1][:, 2])
        print()

        # meanR = np.mean(dataset[:, 1][0])
        # meanG
        # meanB 

        # stdR
        # stdG 
        # stdB
        # np.mean(l), np.std(l)

    # def normalize_val(self):
    #     self.cultural_site_val.normalize_dataset()

    # def normalize_test(self):
    #     self.cultural_site_test.normalize _dataset()

    def train_dataloader(self):
        return DataLoader(self.cultural_site_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cultural_site_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cultural_site_test, batch_size=self.batch_size, num_workers=self.num_workers)