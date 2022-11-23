# deep-learning-project

This is a deep learning project for the academic course "Deep Learning", IT department (DMI), UniCT.

---------------
## Requirements

Install requirements by running (python3 and pip3 required):

```bash
  $ pip3 install -r ./requirements.txt
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

---------------

## Study cases list

**Models:** Each case present ResNet18 and ResNet50 models training

1. Filtering dataset
    1. Training performed on the unfiltered dataset
    2. Training performed on the filtered dataset
2. Domain Adaptation
    1. Baseline approaches without adaption
        1. The model is trained on synthetic images and tested on synthetic images (no domain shift)
        2. The model is trained on synthetic images and tested on real images (no adaptation)
        3. The model is trained on real images and tested on real images (Oracle)
    2. Domain adaptation through image-to-image translation (CycleGAN)
        1. CycleGAN training
        2. Testing CycleGAN and ResNet (real->synthetic->inference)
        3. Training ResNet with CycleGAN translated images (synthetic->real->training)
    3. Feature-level Domain Adaptation
        1. Gradient Reversal Layer
        2. Adversarial domain adatation (ADDA)
    4. Combining Image-to-image translation and Feature-level domain adaptation
        1. Training ResNet (CycleGAN -> Gradient Reversal Layer -> ResNet)
        2. Training ResNet (CycleGAN -> ADDA -> ResNet)