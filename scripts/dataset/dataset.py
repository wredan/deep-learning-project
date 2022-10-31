from typing import Callable, Optional
import json
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import os
import numpy as np

class CulturalSiteDataset(VisionDataset):

    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    REAL_DATASET = 0
    SYNTHETIC_DATASET = 1    

    REAL = "real"
    SYNTHETIC = "syntehtic"

    def __init__(self, dataset_base_path, dataset_stage=TRAIN, dataset_type=SYNTHETIC_DATASET, transform: Optional[Callable] = None) -> None:
        data_domain = CulturalSiteDataset.REAL if dataset_type == CulturalSiteDataset.REAL_DATASET else CulturalSiteDataset.SYNTHETIC
        if dataset_stage == CulturalSiteDataset.TRAIN:
            dataset_folder = os.path.join(dataset_base_path, data_domain, 'training', 'data')
            labels_file = os.path.join(dataset_base_path, data_domain, 'training', 'labels.json')
        elif dataset_stage == CulturalSiteDataset.VALIDATION and dataset_type != CulturalSiteDataset.REAL_DATASET:
            dataset_folder = os.path.join(dataset_base_path, data_domain, 'validation', 'data')
            labels_file = os.path.join(dataset_base_path, data_domain, 'validation', 'labels.json')
        elif dataset_stage == CulturalSiteDataset.TEST or dataset_type == CulturalSiteDataset.REAL_DATASET:
            dataset_folder = os.path.join(dataset_base_path, data_domain, 'test', 'data')
            labels_file = os.path.join(dataset_base_path, data_domain, 'test', 'labels.json') 
        super().__init__(root = dataset_folder, transform = transform, target_transform=None)

        self.image_dataset = [] # [filename, img, id_class]
        self._load_images_path(dataset_folder)
        self._load_image_classes(labels_file, os.path.join(os.getcwd(), "utils", "image_classes.json"))
        self.image_dataset = np.array(self.image_dataset, dtype=object)

    def get_image_dataset(self):
        return self.image_dataset

    def set_image_dataset(self):
        return self.image_dataset

    def __check_filter_size(self, img_path, soglia_pixel):
        w, h = Image.open(img_path).size
        return w > soglia_pixel and h > soglia_pixel

    def filter_dataset(self, soglia_pixel):
        tmp = [x for x in self.image_dataset if self.__check_filter_size(x[1], soglia_pixel)]
        self.image_dataset = np.array(tmp)

    def resize_dataset(self, min_size):  
        resized_dataset = []     
        for el in self.image_dataset:
            img = Image.open(el[1])
            aspect = img.width / img.height if img.width > img.height else img.height / img.width               
            el[1] = np.asarray(img.resize((int(min_size * aspect), min_size), Image.Resampling.BICUBIC) if img.width > img.height else img.resize((min_size, int(min_size * aspect)), Image.Resampling.BICUBIC))
            resized_dataset.append(el)
        self.image_dataset = np.array(resized_dataset)

    def _load_images_path(self, path):
        suffix = ".jpg"
        for filename in os.listdir(path):
            self.image_dataset.append([filename.removesuffix(suffix), os.path.join(path, filename), None ])

    def _load_image_classes(self, labels_path, class_ids_path):
        labels_content = json.load(open(labels_path))
        ids_content = json.load(open(class_ids_path))
        for img in self.image_dataset:
            for class_el in ids_content["categories"]:
                if class_el["name"] == labels_content["labels"][img[0]]:
                     img[2] = class_el["id"]

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index: int):
        img_path = self.image_dataset[index][1]
        image_class = self.image_dataset[index][2]
        img = Image.open(img_path)
        if self.transform is not None:
            transf_img = self.transform(img)
        return transf_img, image_class

    def __len__(self) -> int:
        return len(self.image_dataset)