from scripts.dataset.GAN.dataset_GAN import CulturalSiteDatasetGAN
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Global Variable
BASE_CLASS_DATASETS_PATH = os.path.join(os.getcwd(), 'CLASS-EGO-CH-OBJ-ADAPT')

class CulturalSiteDataModuleGAN(pl.LightningDataModule):

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
        if stage==CulturalSiteDataModuleGAN.FIT_STAGE or stage==CulturalSiteDataModuleGAN.ALL_STAGE:
            train_transform = transforms.Compose([
                transforms.Resize(256, Image.BICUBIC), #ridimensioniamo a una dimensione più grande di quella di input
                transforms.RandomCrop(224), #random crop alla dimensione di input
                transforms.RandomHorizontalFlip(), #random flip orizzontale
                transforms.ToTensor(), #trasformiamo in tensore
                #? applichiamo la normalizzazione (detto dal prof furnari: solitamente quando si usano le GAN non si fanno grosse assunzioni, 
                #? va bene lasciare una media e std intorno ai volori 0.5 sotto proposti)
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

            self.cultural_site_train = CulturalSiteDatasetGAN(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                transform= train_transform,
                mode=CulturalSiteDatasetGAN.TRAIN)

        if stage==CulturalSiteDataModuleGAN.TEST_STAGE or stage==CulturalSiteDataModuleGAN.ALL_STAGE:
            test_transform = transforms.Compose([
                transforms.Resize(256, Image.BICUBIC),
                transforms.CenterCrop(224),                                                         
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

            self.cultural_site_test = CulturalSiteDatasetGAN(
                dataset_base_path=BASE_CLASS_DATASETS_PATH,
                transform= test_transform,
                mode=CulturalSiteDatasetGAN.TEST)

    def filter_train(self, pixel_threshold):
        self.cultural_site_train.filter_dataset(pixel_threshold)

    def filter_test(self, pixel_threshold):
        self.cultural_site_test.filter_dataset(pixel_threshold)
    
    def train_dataloader(self):
        return DataLoader(self.cultural_site_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.cultural_site_test, batch_size=self.batch_size, num_workers=self.num_workers)