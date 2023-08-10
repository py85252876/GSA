# Gradient attack based on Subsampling and Aggregation

This module contains the requisite code for implementing gradient attacks on the **DDPM** and **Imagen**. We posit that this novel white-box MIA attack method, leveraging model gradients, can lead to enhanced attack efficiency and effectiveness, and is also highly pertinent to real-world scenarios.

This repository contains:

1. Procedures for preprocessing datasets and training both the [DDPM](DDPM/) and [Imagen](Imagen/) .
2. Codes for extracting gradient information from trained [DDPM](DDPM/) and [Imagen](Imagen/) .
3. Approaches to utilizing the extracted gradient data to evaluate attack performance under various metrics, including **Accuracy**, **AUC**, and **TPR**.

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

Same with prepare the **DDPM** dependencies, download [dependencies file](Imagen/requirements.txt) and run it with the same command.

```bash
pip install -r requirements.txt
```

## Prepare Datasets

> If using the CIFAR-10 dataset, you need to run the [prepare_cifar.py](DDPM/prepare_cifar.py) in advance for processing.


To generate datasets for **DDPM's** shadow and target models, execute the script [process_ddpm_ds.py](DDPM/process_ddpm_ds.py) using the

```bash
python process_datasets.py --dataset_dir dataset_dir --output_dir output_dataset_dir --datanum_target_model 30000 --datanum_per_shadow_model 30000 --number_of_shadow_model 5
```

command. In this command, we build 5 shadow model datasets and one target model dataset. Each dataset contains two subsets: member sets and non-member sets. Each dataset contains 30,000 images.

### Prepare Caption

Given that the Imagen model operates as a text-to-image model, it takes a text-image pair as input. However, converting each batch of text into text embeddings every time introduces redundant computations and wastes time. Therefore, for more efficient training, it's advisable to pre-convert captions into embeddings and then input the embedding-image pair for training.

Using the following command to build the embedding file.

```bash
python process_caption.py 
```
Then, same as how to preocess **DDPM** shadow/target models datasets, use 

```bash
python prepare_data.py --embedding_dir embedding_file_dir --output_dir output_dataset_dir
```

## Model Training

To train the **DDPM** model, utilize the command 

```bash
accelerate launch --gpu_ids 0 train_unconditional.py --train_data_dir= train_data_dir --resolution=32 --output_dir=output_model_dir --train_batch_size=32 --num_epochs=400 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no --save_model_epochs=50
```
 Similarly, to train the **Imagen** model, execute 

```bash
python train_model_coco.py --model_dir=output_model_dir --data_dir= train_data_dir --project_name="project_name" --load_train_embedding=embedding_dir --from_scratch=0 --checkpoint_path='None'
```

## Generate Gradient

In the paper, two attack strategies are introduced. To employ the **$\ensuremath{\mathsf{GSA_1}}\xspace$** approach, one can set `attack_method=1`. For executing attacks using the **$\ensuremath{\mathsf{GSA_2}}\xspace$** method, the parameter `attack_method` should be designated as `2`. The default attack method is **$\ensuremath{\mathsf{GSA_1}}\xspace$**. For the **DDPM** model, one can execute [gen_l2_gradients_ddpm.py](DDPM/gen_l2_gradients_ddpm.py). 
```bash
accelerate launch --gpu_ids 0 gen_l2_gradients_ddpm.py   --train_data_dir= train_data_dir  --resolution=64   --model_dir= model_dir   --resume_from_checkpoint="latest"  --which_l2=-1 --output_name= output_gradient_dir --attack_method=1
```
To extract gradients from the **Imagen** model, run [gen_l2_gradients_imagen.py](Imagen/gen_l2_gradients_imagen.py).

```bash
python gen_l2_gradients_imagen.py --gradient_path= output_gradient_dir --data_dir= train_data_dir  --load_train_embedding= embedding_file --checkpoint_path= model_dir --get_unet=1 --attack_method=1 
```
## Test Accuracy

We aimed to conduct a comprehensive evaluation of the effectiveness of our attacks. Consequently, we employed **Accuracy**, **AUC**, as well as **TPR** values at fixed **FPRs** of **1%** and **0.1%** as our evaluation metrics. In [test_attack_accuracy.py](test_attack_accuracy.py) train the `Attack Model` and demonstrate its performance across these diverse evaluation matrices.

```bash
python test_attack_accuracy.py \
--target_model_member_path target_member_gradient_file \
--target_model_non_member_path target_non_member_gradient_file \
--shadow_model_member_path \
    shadow_member_gradient_file \
--shadow_model_non_member_path \
    shadow_non_member_gradient_file
```











