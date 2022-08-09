# deep-learning-project

This is a deep learning project for the academic course "Deep Learning", IT department (DMI), UniCT.

---------------
## Requirements

Install requirements by running (python3 and pip3 required):

```bash
  $ pip3 install -r requirements.txt
```

Get the base dataset [here](https://iplab.dmi.unict.it/EGO-CH-OBJ-ADAPT/EGO-CH-OBJ-ADAPT.zip)

---------------

## Using scripts

To analyze the dataset, run the following script:

```
  $ python3 scripts/show_dataset.py <path_to_COCO_dataset>    
```

To extract patches from the datasets, run the following script (it might take a while):

```
  $ python3 scripts/extract_patches.py <base_path_to_datasets> <new_base_path_to_patches> 
```