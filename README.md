# MIA-Gradient-based-Attack

The `gen_l2_gradients.py` script shows how to calculate the L2 norm of gradients.

The `preprocess_datasets.py` script prepares the dataset for training.

If you want to use the CIFAR-10 dataset, run `prepare_cifar.py` before running preprocess_datasets.py.

## Running locally with PyTorch
### Installing the dependencies

Before running the codes, make sure install all dependencies.

Install requirements.txt and run:

```bash
pip install -r requirements.txt
```

### Prepare the training dataset

The `preprocess_datasets.py` script prepares the dataset for training.

**___Note:
If you want to use the CIFAR-10 dataset, run `prepare_cifar.py` before running preprocess_datasets.py.**

Then cd in the scripts, run `process_data.sh`:

```bash
bash process_data.sh
```

### Train the Target model and Shadow models

run `train_model.sh`:

```bash
bash train_model.sh
```

### After obtaining the gradient information, the L2 norm of the gradient is computed.

run `gen_l2.sh`:

```bash
bash gen_l2.sh
```

### Next, the attack success rate is computed using XGBoost.

run `test_accuracy.sh`:

```bash
bash test_accuracy.sh
```

