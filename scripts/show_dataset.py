import fiftyone as fo
import sys

if len(sys.argv) < 2:
    print("\nThis script gets al least 1 parameters: the path of the dataset (train, test, validation)\n")
    sys.exit(0)

# The directory containing the source images
image_path = sys.argv[1]

# The path to the COCO labels JSON file
labels_path = sys.argv[1] + "/annotations.json"

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
