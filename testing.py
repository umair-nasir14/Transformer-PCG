import pytorch_generative as pg_nn
from pytorch_generative import models

import os
import urllib

import PIL
import numpy as np
import torch
from torch import distributions
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import utils
from torchvision.datasets import vision


def _dynamically_binarize(x):
    return distributions.Bernoulli(probs=x).sample()


def _resize_to_32(x):
    return F.pad(x, (2, 2, 2, 2))

def get_mnist_loaders(batch_size, dynamically_binarize=False, resize_to_32=False):
    """Create train and test loaders for the MNIST dataset.
    Args:
        batch_size: The batch size to use.
        dynamically_binarize: Whether to dynamically  binarize images values to {0, 1}.
        resize_to_32: Whether to resize the images to 32x32.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = [transforms.ToTensor()]
    if dynamically_binarize:
        transform.append(_dynamically_binarize)
    if resize_to_32:
        transform.append(_resize_to_32)
    transform = transforms.Compose(transform)
    train_loader = data.DataLoader(
        datasets.MNIST("/tmp/data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    test_loader = data.DataLoader(
        datasets.MNIST("/tmp/data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )
    return train_loader, test_loader


    from PIL import Image
import numpy as np

images = []
for i in range(0,500):
    image = torch.from_numpy(np.array(Image.open('/home/munasir/image-gpt/mazes/maze_'+str(i)+'_generation.png').convert('RGB')))
    images.append(image)

class Images(data.Dataset):  
    def __init__(self,control_code, gpt2_type="gpt2", max_length=1024):

        #self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        #self.lyrics = []

        '''for row in df['Lyric']:
          self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))'''
        self.images = []
        for image in images:
            self.images.append(torch.tensor(
                #self.tokenizer.encode(f"<|{[0,0,0]}|>{image[:max_length]}")#<|endoftext|>")
            ))              
        #if truncate:
        #    self.lyrics = self.lyrics[:20000]
        #self.lyrics_count = len(self.lyrics)
        self.images_count = len(self.images)
        
    def __len__(self):
        return self.images_count

    def __getitem__(self, item):
        return self.images[item]
#print(images[0][0][0].shape)
dataset = Images(images,gpt2_type="distilgpt2")    
