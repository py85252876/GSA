# Gradient attack based on Subsampling and Aggregation

This module contains the requisite code for implementing gradient attacks on the DDPM and Imagen models. We posit that this novel white-box MIA attack method, leveraging model gradients, can lead to enhanced attack efficiency and effectiveness, and is also highly pertinent to real-world scenarios.

This repository contains:

1. Procedures for preprocessing datasets and training both the [DDPM](DDPM/) and [Imagen](Imagen/) .
2. Codes for extracting gradient information from trained [DDPM](DDPM/) and [Imagen](Imagen/) .
3. Approaches to utilizing the extracted gradient data to evaluate attack performance under various metrics, including Accuracy, AUC, and TPR.

## Table of Contents

- [Download Dependencies](#download-dependencies)
	- [DDPM dependencies](#ddpm-dependencies)
	- [Imagen dependencies](#imagen-dependencies)
- [Prepare Datasets](#prepare-datasets)
	- [Prepare Caption](#prepare-caption)
- [Model Training](#model-training)
- [Generate Gradient](#generate-gradient)
- [Test Accuracy](#test-accuracy)

## Download Dependencies
### DDPM dependencies

> Before running the code, make sure to install all dependency files.

Install [requirements.txt](DDPM/requirements.txt) and run:

```bash
pip install -r requirements.txt
```

### Imagen dependencies

Same with prepare the DDPM dependencies, download [dependencies file](Imagen/requirements.txt) and run it with the same command.

```bash
pip install -r requirements.txt
```

## Prepare Datasets

> If using the CIFAR-10 dataset, you need to run the [prepare_CIFAR.py](DDPM/prepare_CIFAR.py) in advance for processing.


To generate datasets for DDPM's shadow and target models, execute the script [process_DDPM_ds.py](DDPM/process_DDPM_ds.py) using the

```bash
python process_datasets.py --dataset_dir dataset_dir --output_dir output_ds_dir --datanum_target_model 30000 --datanum_per_shadow_model 30000 --number_of_shadow_model 5
```

command. In this command, we build 5 shadow model datasets and one target model dataset. Each dataset contains two subsets: member sets and non-member sets. Each dataset contains 30,000 images.

### Prepare Caption

Given that the Imagen model operates as a text-to-image model, it takes a text-image pair as input. However, converting each batch of text into text embeddings every time introduces redundant computations and wastes time. Therefore, for more efficient training, it's advisable to pre-convert captions into embeddings and then input the embedding-image pair for training.

Using the following command to build the embedding file.

```bash
python process_caption.py 
```
Then, same as how to preocess DDPM shadow/target models datasets, use 

```bash
python prepare_data.py --embedding_dir embedding_file_dir --output_dir output_dataset_dir
```







