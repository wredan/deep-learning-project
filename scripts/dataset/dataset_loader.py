import os
import zipfile
from scripts.extract_patches import *
import progressbar
import wget 

bar = progressbar.ProgressBar(maxval=7240124142, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

def bar_progress(current, total, width=80):
  bar.update(current)

class CulturalSiteDatasetsLoader():

    def __init__(self, download_path, main_dataset_path, class_dataset_path, zip_file_path):    
        self._class_path_datasets = class_dataset_path
        if not self._classification_datasets_exists():
            print("Classification dataset not found. Checking main dataset...")
            self._main_path_datasets = main_dataset_path
            if not self._main_datasets_exists():
                print("Main dataset not found. Checking zip file dataset...")
                self._get_main_datasets(download_path, zip_file_path) 
            else: print("Main dataset found.")
            print("Start extracting patches...")
            self._extract_patches(self._main_path_datasets, self._class_path_datasets)  
            print("Patches extracted successfully!")  
        else: print("Classification dataset found.")

    def _get_main_datasets(self, download_path, zip_file_path): # download and extract dataset
        if not self._main_datasets_zip_exists(zip_file_path):
            print("Zip file dataset not found. Pulling from resource (", download_path, ") ...")
            bar.start()
            wget.download(download_path, os.path.join(os.getcwd(), ""), bar=bar_progress)
            bar.finish()
        self.extract_file(zip_file_path)
        
    def extract_file(self, zip_file_path):  
        print("Zip file found, start unzipping...")
        save_path = os.path.join(self._main_path_datasets, "")
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(save_path)     
        print("File unzipped successfully!")

    def _extract_patches(self, main_path, save_path):
        print(main_path, save_path)
        ExtractPatches(main_path, save_path).extract()

    def _main_datasets_exists(self):
        return os.path.isdir(self._main_path_datasets)

    def _main_datasets_zip_exists(self, zip_file_path):
        return os.path.isfile(zip_file_path)

    def _classification_datasets_exists(self):
        return os.path.isdir(self._class_path_datasets)