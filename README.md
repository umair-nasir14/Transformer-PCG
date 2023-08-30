# Transformer-PCG

This repository contains code to reproduce all of our experiments.
The file structure for important files are as follows:
```
├── Mario                    -> Dataset for Mario levels
├── ckpts                    -> Checkpoints for trained models
├── notebooks                -> Notebooks demonstrating experiments with different models
├── pytorch_generative       -> Code for generative AI with PyTorch
├── wandb                    -> Weights and Biases for tracking models performance and hyperparameters
├── Datasets.py              -> Create dataset for our transformer model
├── train.py                 -> To train your model
```
To create the environment, use `requirements.txt`
```
pip install -r requirements.txt
```
To reproduce the experiments, run 
```
python train.py
```
