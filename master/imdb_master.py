"""
  FileName     [ imdb.py ]
  PackageName  [ final ]
  Synopsis     [ Dataloader of IMDb dataset ]
  Structure:
    IMDb/
    |---val/
    |   |---<movie_name>/
    |   |   |---candidates/
    |   |   |   |---<id>.jpg
    |   |   |---cast/
    |   |   |   |---<id>.jpg
    |   |   |---cast.json
    |   |   |---candidate.json
    |---|---<movie_name>/
    ...
"""

import csv
import itertools
import os
import pprint
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import utils

class IMDbTrainset(Dataset):
    def __init__(self, movie_path, feature_path, label_path, mode, cast_image=True, keep_others=True, transform=None, debug=False):
        assert (movie_path is not None), "movie_path is needed for IMDbDataset"
        assert ((mode == 'classify') or (mode == 'features') or (mode == 'faces')), "The parameter 'mode' must be 'classify', 'feature' or 'faces'"
        assert (not (mode == 'features') or (cast_image)), "Cast images must be loaded in dataset if mode is 'features'"

        self.movie_path = movie_path
        self.root_path  = os.path.dirname(self.movie_path)
        
        self.mode = mode
        self.keep_others = keep_others
        self.cast_image  = cast_image
        self.debug = debug

        self.transform = transform

        self.movies         = sorted(os.listdir(self.movie_path))
        self.candidate_json = [pd.read_json(os.path.join(self.movie_path, filename, 'candidate.json'), orient='index', typ='series') 
                                  for filename in self.movies]
        self.cast_json      = [pd.read_json(os.path.join(self.movie_path, filename, 'cast.json'), orient='index', typ='series') 
                                  for filename in self.movies]

        # Read as pandas.DataFrame and make it
        self.candidates = pd.concat(self.candidate_json, axis=0, keys=self.movies).reset_index()
        self.casts      = pd.concat(self.cast_json, axis=0, keys=self.movies).reset_index()

        if not keep_others:
            self.candidates = self.candidates[self.candidates[0] != "others"]

        # Add 'others' with the label query table only, don't add to image query table
        if cast_image:
            self.images = pd.concat((self.candidates, self.casts), axis=0, ignore_index=True)
        
        if keep_others:
            num_casts = self.casts.shape[0]
            self.casts.loc[num_casts] = ['others', 'no_others_exists.jpg', 'others']

        self.classes = list(self.casts['level_0'])

        # print(self.candidates.columns)  # ['level_0', 'level_1', 0]
        # print('level_0 :\n', self.candidates[self.candidates[0] == 'others')

        # Total images in dataset
        # print("Total candidates in dataset: {}".format(self.candidates.shape))
        # Without "others"
        # print("Total casts in dataset:      {}\n".format(self.casts.shape))
        # print("Total casts in unique: {}".format(self.casts.shape))

    @property
    def num_casts(self):
        return self.casts.shape[0]

    def __len__(self):
        if self.cast_image:
            return self.images.shape[0]

        if self.mode == 'classify' or self.mode == 'faces':
            return self.candidates.shape[0]

    def __getitem__(self, index):
        # -------------------------------------------------
        # Mode:
        #   Classify: 
        #     get 1 image and 1 label
        #   Faces: get 1
        #     get 1 image, label is 1 if it contains a face
        #   Features:
        #     get 1 image only. 
        # -------------------------------------------------
        if self.cast_image:
            image_path, cast = self.images.iat[index, 1], self.images.iat[index, 2]

        elif self.mode == 'classify' or self.mode == 'faces':
            image_path, cast = self.candidates.iat[index, 1], self.candidates.iat[index, 2]

        # ---------------------------------------------------
        # To Read the images
        # ---------------------------------------------------
        image = Image.open(os.path.join(self.root_path, image_path))

        # ------------------------------------------------- # 
        # Label Output dimension:  (1)                      #
        # Images Output dimension: (channel, height, width) #
        # ------------------------------------------------- #
        if self.transform:
            image = self.transform(image)

        # string label >> int label
        if self.mode == 'classify':
            label_mapped = self.casts.index[self.casts[0] == cast].tolist()[0] 
            # total : 1 element 
        
            if self.debug:
                print("label_mapped : {} <--> {}".format(label_mapped, cast))
        
        if self.mode == 'faces':
            label_mapped = (cast != 'others')

            if self.debug:
                print("label_mapped : {} <--> {}".format(label_mapped, cast))
            
        if self.mode == 'features':
            label_mapped = self.casts.index[self.casts[0] == cast].to_list()[0]

            if self.debug:
                print("label_mapped : {} <--> {}".format(label_mapped, cast))
	
#        print(image.size())
        return image, label_mapped

class IMDbFolderLoader(Dataset):
    def __init__(self, movie_path, transform=None, debug=False):
        self.movie_path = movie_path
        self.transform  = transform

        # Read as pandas.DataFrame
        self.candidates = pd.read_json(os.path.join(self.movie_path, 'candidate.json'), orient='index', typ='series').reset_index() 
        self.casts      = pd.read_json(os.path.join(self.movie_path, 'cast.json'), orient='index', typ='series') .reset_index()
        self.images     = pd.concat((self.candidates, self.casts), axis=0, ignore_index=True)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # --------------------------------------------- #
        # To Read the images:                           #
        #   Output dimension: (channel, height, width)  #
        # --------------------------------------------- #
        image_path, cast = self.images.iat[index, 0], self.images.iat[index, 1]
        image_path = os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path))
        image = Image.open(os.path.join(self.movie_path, image_path))

        if self.transform:
            image = self.transform(image)

        return image, 0

def dataloader_unittest(path, debug=False):
    print("Classify setting (keep others): ")
    
    dataset = IMDbTrainset(
        movie_path = path,
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode = 'classify',
        cast_image=False,
        keep_others=True,
        debug = debug,
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image, label) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        print("Label.shape: {}".format(label.shape))
        # print("Label: {}".format(label))
        print()

        # if "others" in label:
        # if 198 in label:   # "others" mapped to 198
        #     print('dataloader unitest finished, has 198("others") in labels.')
        break

    # ------------------------------------------------------------------------------ #

    print("Classify setting (remove others): ")

    dataset = IMDbTrainset(
        movie_path = path,
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode = 'classify',
        cast_image=True,
        keep_others=False,
        debug = debug,
        transform = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image, label) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        print("Label.shape: {}".format(label.shape))
        # print("Label: {}".format(label))
        print()

        break

    # ------------------------------------------------------------------------------ #
                                                                                                                                                        
    print("Features setting: ")

    dataset = IMDbTrainset(
        movie_path = path,
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode='features',
        keep_others=True,
        cast_image=True,
        debug = debug,
        transform = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image, _) in enumerate(dataloader, 1):
        print(image.size())
        print("Image.shape: {}".format(image.shape))
        print()

        break

    # ------------------------------------------------------------------------------ #
    print("Faces setting: ")

    dataset = IMDbTrainset(
        movie_path = path,
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode = 'faces',
        cast_image=False,
        debug = debug,
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image, label) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        print("Label.shape: {}".format(label.shape))
        print("Label: {}".format(label))
        print()
        break


if __name__ == "__main__":
    
    path = "/media/disk1/EdwardLee/dataset/IMDb/val"
    dataloader_unittest(path, True)
