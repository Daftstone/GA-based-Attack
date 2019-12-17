# Hybrid of Genetic Algorithm and Fast Gradient Sign Method for Generating Adversarial Examples

This project is for the paper "Hybrid of Genetic Algorithm and Fast Gradient Sign Method for Generating Adversarial Examples". Some codes are from cleverhans.

The code was developed on Python 3.6


## 1. Install dependencies.
Our experiment runs on GPU,, install this list:
```bash
pip install -r requirements_gpu.txt
```

## 2. Download dataset and pre-trained models.
Download [dataset](https://drive.google.com/file/d/1gyBeIpy4WzO17_7hjZKR6flP9oPhHVfU/view) 
and extract it to the root of the program, which contains the MNIST, FMNIST, and SVHN dataset.

Download [pre-trained models](https://drive.google.com/file/d/1kYQC28-FNDhMOx8k-mLji20dkTrCLm6F/view?usp=sharing)
and extract it to the root of the program. 
## 3. Usage of `python attack.py`
```
usage: python main.py [--dataset DATASET_NAME] [--solve ATTACK_NAME]
               [--eps perturbation_step] [--is_train [IS_TRAIN]]
               [--nb_epochs EPOCHS_NUMBER] [--n_point POINT]
               [--generation GENERATION]

optional arguments:
  --dataset DATASET_NAME
                        Supported: mnist, fmnist, svhn, cifar10.
  --solve ATTACK_NAME
                        Supported: GA, Hybrid-GA, MF-GA, Hybrid-MF-GA.
  ----eps DETECTION_NAME
                        Perturbation step.
  --is_train [IS_TRAIN]
                        User this parameter to train online, otherwise remove the parameter.
  ----nb_epochs EPOCHS_NUMBER
                        Number of epochs the classifier is trained.
  --n_point POINTS
                        Number of individuals in the population
  --generation GENERATION
                        the number of epochs of training
```

### 4. Example.
Use pre-trained model.
```bash
python main.py --dataset mnist --solve GA --eps 0.03
```
Train model online.
```bash
python main.py --dataset mnist --solve GA --eps 0.03 --is_train --nb_epochs 100
```

