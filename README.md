#Gradient attack based on Subsampling and Aggregation

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

> Before running the codes, make sure install all dependencies.

Install [requirements.txt](DDPM/requirements.txt) and run:

```bash
pip install -r requirements.txt
```

### Imagen dependencies

Same with prepare the DDPM dependencies, download [dependencies](Imagen/requirements.txt) file and run it with the same command.

```bash
pip install -r requirements.txt
```



