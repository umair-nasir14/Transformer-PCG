import torch
import PIL
from PIL import Image
import numpy as np
from torch.utils import data
from torchvision import transforms

from common import ToTensor

import os


class MazeDataset(data.Dataset):
    """Maze dataset."""

    def __init__(self,train=True, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            train:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.train = train
        
        self.images = []
        if self.train:
            
            for i in range(0,1586):
                image = Image.open('/home/munasir/image-gpt/mazez/pymaze_windows/mazes/maze_'+str(i)+'_generation.png').convert('L')
                self.images.append(image)
        else:
            
            for i in range(0,10):
                image = Image.open('/home/munasir/image-gpt/mazez/pymaze_windows/extra_mazes/maze_'+str(i)+'_generation.png').convert('L')
                self.images.append(image)
        
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.train:
            path_1 = '/home/munasir/image-gpt/mazez/pymaze_windows/mazes/maze_'+str(idx)+'_generation.png'
        else:
            path_1 = '/home/munasir/image-gpt/mazez/pymaze_windows/extra_mazes/maze_'+str(idx)+'_generation.png'

        
        image = Image.open(path_1)
        image = image.convert('RGB')
        image = np.array(image)
        image = np.where(image>128,[255],[0])
        
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample


train_set = MazeDataset(train=True, transform=None)
train_eval_set = MazeDataset(train=False, transform=None)


def get_maze_loaders(batch_size):
    """Create train and test loaders for the MAZE dataset.
    Args:
        batch_size: The batch size to use.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=os.cpu_count()-1,
    )
    train_eval_loader = data.DataLoader(
        train_eval_set,
        batch_size=batch_size,
        #num_workers=os.cpu_count()-1,
    )
    return train_loader, train_eval_loader








########################################################################################






class MarioDataset(data.Dataset):
    """Mario dataset."""

    def __init__(self,train=True, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            train:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.transform = transform
        self.train = train
        
        self.images = []
        if self.train:
            
            for i in range(0,999):
                image = Image.open('/home/munasir/image-gpt/igpt-pytorch/pytorch-generative/Mario/n-gram_'+str(i)+'.png')
                self.images.append(image)
        else:
            
            for i in range(3,5):
                image = Image.open('/home/munasir/image-gpt/igpt-pytorch/pytorch-generative/Mario/n-gram_'+str(i)+'.png')
                self.images.append(image)
        
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.train:
            path_1 = '/home/munasir/image-gpt/igpt-pytorch/pytorch-generative/Mario/n-gram_'+str(idx)+'.png'
        else:
            path_1 = '/home/munasir/image-gpt/igpt-pytorch/pytorch-generative/Mario/n-gram_'+str(idx)+'.png'

        
        image = Image.open(path_1)
        image = image.convert('RGB').resize((200,14))
        image = np.array(image)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample

train_set = MarioDataset(train=True, transform=transforms.Compose([ToTensor()]))
train_eval_set = MarioDataset(train=False, transform=transforms.Compose([ToTensor()]))




def get_mario_loaders(batch_size):
    """Create train and test loaders for the MARIO dataset.
    Args:
        batch_size: The batch size to use.
    Returns:
        Tuple of (train_loader, test_loader).
    """
    
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=os.cpu_count()-1,
    )
    train_eval_loader = data.DataLoader(
        train_eval_set,
        batch_size=batch_size,
        #num_workers=os.cpu_count()-1,
    )
    return train_loader, train_eval_loader

