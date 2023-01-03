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
    
    # Find training data
    files = os.listdir(input_filepath)
    data_all = [
        np.load(os.path.join(input_filepath, f))
        for f in files
        if f.endswith(".npz") and "train" in f
    ]
    print(len(data_all))
   
    # Combine .npz files
    keys = ['images', 'labels']
    
    merged_data = dict.fromkeys(keys, None)
    merged_data = dict(data_all[0])
    
    #print(merged_data['images'].shape)
    #print((data_all.shape))
    for data in data_all[1:]:
        for k in data.keys():
            merged_data[k] = np.vstack((merged_data[k], dict(data)[k]))
    print(merged_data['labels'].size)
    merged_data["labels"] = np.reshape(
        merged_data["labels"], merged_data["labels"].size
    )
    print(merged_data['labels'].shape)

    np.savez(os.path.join(output_filepath, "train_data_raw_merged.npz"), **merged_data)

    # Load in the train file
    train = np.load(os.path.join(output_filepath, "train_data_raw_merged.npz"))


    # Normalizing scheme 
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    
    # Transform the data
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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
