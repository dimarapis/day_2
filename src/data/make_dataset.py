# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from shutil import copyfile


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train_data = [ ]
    for i in range(5):
        train_data.append(np.load(os.path.join(input_filepath,f"train_{i}.npz"), allow_pickle=True))
    
    train_images = torch.tensor(np.concatenate([c['images'] for c in train_data])).reshape(-1, 1, 28, 28)
    train_targets = torch.tensor(np.concatenate([c['labels'] for c in train_data]))
    
    test_data = [ ]

    test_data = np.load(os.path.join(input_filepath,"test.npz"), allow_pickle=True)
    test_images = torch.tensor(test_data['images']).reshape(-1, 1, 28, 28)
    test_targets = torch.tensor(test_data['labels'])
        
  
    # Mean over batch, height and width, but not over the channels
    mean = torch.mean(train_images, dim=[0,2,3])
    channels_squared_sum = torch.mean(train_images**2, dim=[0,2,3])
  

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum - mean ** 2) ** 0.5
    
    print(mean,std)
    # Normalizing scheme 
    transform = transforms.Normalize((mean), (std))

    transformed_train_images = transform(train_images)
    transformed_test_images = transform(test_images)
    
    print(torch.min(transformed_train_images),torch.mean(transformed_train_images),torch.max(transformed_train_images))
    print(torch.min(transformed_test_images),torch.mean(transformed_test_images),torch.max(transformed_test_images))

    
    
    
    #print(transformed_train_images.shape)
    '''    # Transform the data
    images_train = transform(train.f.images).permute(1,2,0)
    labels_train = torch.Tensor(train.f.labels).type(torch.LongTensor)
    
    test = np.load(os.path.join(input_filepath, "test.npz"))
    print(test['images'].shape)
    images_test = transform(test.f.images)
    print(images_test.shape)
    labels_test = torch.Tensor(test.f.labels).type(torch.LongTensor)
    
    print(images_train.shape, len(labels_train), len(images_test), len(labels_test))
    # Save the individual tensors
    #torch.save(images_train, os.path.join(output_filepath, "images_train.pt"))
    #torch.save(labels_train, os.path.join(output_filepath, "labels_train.pt"))

    # Pass test data through to output
    #copyfile(
    #    os.path.join(input_filepath, "test.npz"),
    #    os.path.join(output_filepath, "test.npz"),
    #)
    '''
    
    return dataset_train,dataset_test

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    dataset_train,dataset_test = main()
