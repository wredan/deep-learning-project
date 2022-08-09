import fiftyone as fo

# The directory containing the source images
image_path = "../deep_learning_project/EGO-CH-OBJ-ADAPT/real/training"

# The path to the COCO labels JSON file
labels_path = "../deep_learning_project/EGO-CH-OBJ-ADAPT/real/training/annotations.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=image_path,
    labels_path=labels_path,
    name="EGO-CH-OBJ-ADAPT"
)

session = fo.launch_app(dataset, port=5151)

# Blocks execution until the App is closed
session.wait()
