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

    RESIZE_HEIGHT = 32
    RESIZE_WIDTH = 32

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
        self._load_images(dataset_folder)
        self._load_image_classes(labels_file) # todo: remove hardcode (add config file)
        self._load_class_ids(os.path.join(os.getcwd(), "utils", "image_classes.json")) # todo: remove hardcode o salvare 
        self.image_dataset = np.array(self.image_dataset, dtype=object)

    def get_image_dataset(self):
        return self.image_dataset

    def set_image_dataset(self):
        return self.image_dataset

    def filter_dataset(self, soglia_pixel):
        tmp = [x for x in self.image_dataset if x[1].shape[0] > soglia_pixel and x[1].shape[1] > soglia_pixel]
        self.image_dataset = np.array(tmp)

    def resize_dataset(self):  
        resized_dataset = []     
        for el in self.image_dataset:
            el[1] = np.asarray(Image.fromarray(el[1]).resize((CulturalSiteDataset.RESIZE_WIDTH, CulturalSiteDataset.RESIZE_HEIGHT)))
            resized_dataset.append(el)
        self.image_dataset = np.array(resized_dataset)

    def _load_images(self, path):
        suffix = ".jpg"
        for filename in os.listdir(path):
            im = Image.open(os.path.join(path, filename))
            self.image_dataset.append([filename.removesuffix(suffix), np.asarray(im), None ])

    def _load_image_classes(self, path):
        file = open(path)
        content = json.load(file)
        for el in self.image_dataset:
            el[2] = content["labels"][el[0]]

    def _load_class_ids(self, path):  # TODO: è possibile fare tutto in un unico passaggio nella funzione sopra, da sistemare
        file = open(path)
        content = json.load(file)
        for i in range(len(self.image_dataset)):
            for el in content["categories"]:
                if el["name"] == self.image_dataset[i][2]:
                    self.image_dataset[i][2] = el["id"]

    def __getitem__(self, index: int):
        img = self.image_dataset[index][1]
        image_class = self.image_dataset[index][2]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, image_class

    def __len__(self) -> int:
        return len(self.image_dataset)