
import fiftyone as fo
import sys

if len(sys.argv) < 3:
    print("\nThis script gets al least 2 parameters: first one is base path of the datasets, second one is base path where to save the patches.\n")
    sys.exit(0)

for i in range(0, 5):
    if i == 0:
        image_sub_path = "real/test"
    elif i == 1:
        image_sub_path = "real/training"
    elif i == 2:
        image_sub_path = "syntehtic/test"
    elif i == 3:
        image_sub_path = "syntehtic/training"
    elif i == 4:
        image_sub_path = "syntehtic/validation"

    # The path to the source images
    image_path = sys.argv[1] + image_sub_path

    # The path to the COCO labels JSON file
    labels_path = sys.argv[1] + image_sub_path + "/annotations.json"

    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_path,
        labels_path=labels_path,
        name=image_sub_path
    )

    patches = dataset.to_patches("ground_truth")

    # https://voxel51.com/docs/fiftyone/api/fiftyone.types.dataset_types.html
    # dataset_type=fo.types.FiftyOneImageClassificationDataset, patch dentro la cartella "data" + labels.json
    # dataset_type=fo.types.ImageClassificationDirectoryTree, divide per cartelle le patch in base alla classe
    patches.export(
        export_dir=sys.argv[2] + image_sub_path,
        dataset_type=fo.types.FiftyOneImageClassificationDataset,
        label_field="ground_truth",
    )