
import fiftyone as fo
import os

class ExtractPatches():

    def __init__(self, main_path, save_path):
        self.main_path = main_path
        self.save_path = save_path

    def delete_dataset(self, dataset_name):
        if fo.core.dataset.dataset_exists(dataset_name):
            fo.core.dataset.delete_dataset(dataset_name)

    def extract_database(self, image_sub_path):
         # The path to the source images
        image_path = os.path.join(self.main_path, image_sub_path)
        print(image_path)

        # The path to the COCO labels JSON file
        labels_path = os.path.join(self.main_path, image_sub_path, "annotations.json") 
        print(image_sub_path)
        print()

        # delete if exist
        self.delete_dataset(image_sub_path)

        # Import the dataset
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=image_path,
            labels_path=labels_path,
            name=image_sub_path
        )

        patches = dataset.to_patches("ground_truth")
        patches.export(
            export_dir=os.path.join(self.save_path, image_sub_path),
            dataset_type=fo.types.FiftyOneImageClassificationDataset,
            label_field="ground_truth",
        )

    def extract(self):
        self.extract_database(os.path.join("real", "test"))
        self.extract_database(os.path.join("real", "training"))
        self.extract_database(os.path.join("syntehtic", "training"))
        self.extract_database(os.path.join("syntehtic", "validation"))
        self.extract_database(os.path.join("syntehtic", "test"))      

            

           