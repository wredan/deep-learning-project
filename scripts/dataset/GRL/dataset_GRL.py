import random
from PIL import Image
import os
import glob
from typing import Callable, Optional
from torch.utils.data import Dataset
import json
import numpy as np

class CulturalSiteDatasetGRL(Dataset):

    TRAIN = "training"
    VALIDATION = "test"
    TEST = "validation"

    REAL = "real"
    SYNTHETIC = "syntehtic"

    def __init__(self, dataset_base_path=None, dataset_stage=TRAIN, transform: Optional[Callable] = None) -> None:
            if dataset_base_path is not None:
                self.transform = transform
                if dataset_stage == CulturalSiteDatasetGRL.TRAIN or dataset_stage == CulturalSiteDatasetGRL.TEST:
                    dataset_folder_syn, labels_file_syn, dataset_folder_real, labels_file_real = self.__get_train_or_test_path(dataset_base_path, dataset_stage)
                    self.load_datasets(dataset_folder_syn, labels_file_syn, dataset_folder_real, labels_file_real)
                elif dataset_stage == CulturalSiteDatasetGRL.VALIDATION:
                    dataset_folder_syn = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.SYNTHETIC, 'validation', 'data')
                    labels_file_syn = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.SYNTHETIC, 'validation', 'labels.json')
                    self.load_datasets(dataset_folder_syn, labels_file_syn)
            else:
                self.syn_image_dataset = []
                self.real_image_dataset = []

    def __get_train_or_test_path(self, dataset_base_path, mode):
        dataset_folder_syn = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.SYNTHETIC, '%s' % mode, 'data')
        labels_file_syn = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.SYNTHETIC, '%s' % mode, 'labels.json')
        dataset_folder_real = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.REAL, '%s' % mode, 'data')
        labels_file_real = os.path.join(dataset_base_path, CulturalSiteDatasetGRL.REAL, '%s' % mode, 'labels.json')
        return dataset_folder_syn, labels_file_syn, dataset_folder_real, labels_file_real

    def get_syn_image_dataset(self):
        return self.syn_image_dataset

    def set_syn_image_dataset(self, image_dataset):
        self.syn_image_dataset = np.array(image_dataset)

    def get_real_image_dataset(self):
        return self.real_image_dataset

    def set_real_image_dataset(self, image_dataset):
        self.real_image_dataset = np.array(image_dataset)

    def load_datasets(self, dataset_folder_syn, labels_file_syn, dataset_folder_real=None, labels_file_real=None):
        self.syn_image_dataset = self._load_images_path(dataset_folder_syn)
        self.syn_image_dataset = self._load_image_classes(self.syn_image_dataset, labels_file_syn, os.path.join(os.getcwd(), "utils", "image_classes.json"))
        self.syn_image_dataset = np.array(self.syn_image_dataset, dtype=object)

        if dataset_folder_real:
            self.real_image_dataset = self._load_images_path(dataset_folder_real)
            self.real_image_dataset = self._load_image_classes(self.real_image_dataset, labels_file_real, os.path.join(os.getcwd(), "utils", "image_classes.json"))
            self.real_image_dataset = np.array(self.real_image_dataset, dtype=object)
    
    def _load_images_path(self, path):
        image_dataset = []
        suffix = ".jpg"
        for filename in os.listdir(path):
            image_dataset.append([filename.removesuffix(suffix), os.path.join(path, filename), None ])
        return image_dataset

    def _load_image_classes(self, image_dataset, labels_path, class_ids_path):
        labels_content = json.load(open(labels_path))
        ids_content = json.load(open(class_ids_path))
        for img in image_dataset:
            for class_el in ids_content["categories"]:
                if class_el["name"] == labels_content["labels"][img[0]]:
                     img[2] = class_el["id"]
        return image_dataset

    def __check_filter_size(self, img_path, pixel_threshold):
        w, h = Image.open(img_path).size
        return w > pixel_threshold and h > pixel_threshold

    def filter_dataset(self, pixel_threshold):
        tmp = [x for x in self.syn_image_dataset if self.__check_filter_size(x, pixel_threshold)]
        self.syn_image_dataset = tmp
        tmp = [x for x in self.real_image_dataset if self.__check_filter_size(x, pixel_threshold)]
        self.real_image_dataset = tmp

    def __getitem__(self, index: int):
        _, syn_img_path, syn_img_class = self.syn_image_dataset[index]
        _, real_img_path, _ = self.real_image_dataset[random.randint(0, len(self.target)-1)]
        syn_img = Image.open(syn_img_path)
        real_img = Image.open(real_img_path)
        if self.transform is not None:
            transf_img_syn = self.transform(syn_img)
            transf_img_real = self.transform(real_img)
        return transf_img_syn, transf_img_real, syn_img_class

    def __len__(self) -> int:
        return len(self.syn_image_dataset)