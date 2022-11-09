from scripts.dataset.GRL.dataset_GRL import CulturalSiteDatasetGRL
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

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
        test_val_transform = transforms.Compose([
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),                                                         
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # TODO: rivedere sti valori
        ])

        if stage==CulturalSiteDataModuleGRL.FIT_STAGE or stage==CulturalSiteDataModuleGRL.ALL_STAGE:
            train_transform = transforms.Compose([
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC), #ridimensioniamo a una dimensione più grande di quella di input
                transforms.RandomCrop(224), #random crop alla dimensione di input
                transforms.RandomHorizontalFlip(), #random flip orizzontale
                transforms.ToTensor(), #trasformiamo in tensore
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # TODO: rivedere sti valori
            ])

            self.cultural_site_train = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.TRAIN, 
                transform=train_transform)

            train_set = self.cultural_site_train.get_real_image_dataset()
            new_train_set, new_validation_set = train_test_split(train_set, test_size = 0.1) # splitting train ratio 90/10
            self.cultural_site_train.set_real_image_dataset(new_train_set)
            
            self.cultural_site_val = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.VALIDATION, 
                transform=test_val_transform)
            self.cultural_site_val.set_real_image_dataset(new_validation_set)

        if stage==CulturalSiteDataModuleGRL.TEST_STAGE or stage==CulturalSiteDataModuleGRL.ALL_STAGE:
            self.cultural_site_test = CulturalSiteDatasetGRL(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                dataset_stage=CulturalSiteDatasetGRL.TEST, 
                transform=test_val_transform)

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