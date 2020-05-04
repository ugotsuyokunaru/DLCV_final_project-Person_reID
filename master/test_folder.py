"""
  FileName     [ test.py ]
  PackageName  [ final ]
  Synopsis     [ Test the Person_reID model with the IMDb dataset ]

  Dataset:
  - IMDb

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.

  Pretrain network:
  - PCB:
  - DenseNet:
  - NAS:
  - ResNet: 
"""

from __future__ import division, print_function

import argparse
import csv
import math
import os
import subprocess
import time

import numpy as np
import scipy.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

import evaluate_gpu
import evaluate_rerank
import final_eval
import imdb
import pcb_extractor
import utils
from model import PCB, PCB_test, ft_net, ft_net_dense, ft_net_NAS


def main(opt):
    if not opt.features:
        # --------------------------------- #
        # Load configuration of this model  #
        # --------------------------------- #
        config_path = os.path.join(os.path.dirname(opt.resume), 'opts.yaml')
        with open(config_path, 'r') as stream:
            config = yaml.load(stream)

        opt.name = config['name']
        opt.PCB       = config['PCB']
        opt.use_dense = config['use_dense']
        opt.use_NAS   = config['use_NAS']
        opt.stride    = config['stride']
        opt.img_size  = config['img_size']

        if 'nclasses' in config: # tp compatible with old config files
            opt.nclasses = config['nclasses']
        else: 
            opt.nclasses = 199 # Including "others"

        opt.img_size = tuple(opt.img_size)
        opt.gpu_ids = list(filter(lambda x: x >= 0, opt.gpu_ids))
        ms = [math.sqrt(float(s)) for s in opt.ms]

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            cudnn.benchmark = True

        # set gpu / cpu
        use_gpu = torch.cuda.is_available()

        # Load datas with dataloader.
        if not opt.PCB:
            data_transforms = transforms.Compose([
                    transforms.Resize(opt.img_size, interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if opt.PCB:
            data_transforms = transforms.Compose([
                transforms.Resize(opt.img_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])

        # Load trained model 
        if opt.use_dense:
            model_structure = ft_net_dense(opt.nclasses)
        elif opt.use_NAS:
            model_structure = ft_net_NAS(opt.nclasses)
        else:
            model_structure = ft_net(opt.nclasses, stride = opt.stride)

        if opt.PCB:
            model_structure = PCB(opt.nclasses)

        model = utils.load_network(model_structure, opt.resume)

        # Remove the final fc layer and classifier layer
        if opt.PCB:
            model = PCB_test(model)
        else:
            model.classifier.classifier = nn.Sequential()

        # Change to test mode
        model = model.eval()

        # Throught it to the gpu
        if use_gpu:
            model = model.cuda()
        
        for movie in tqdm(sorted(os.listdir(opt.testset))):
            image_datasets = imdb.IMDbFolderLoader(
                movie_path=os.path.join(opt.testset, movie), 
                transform=data_transforms
            )
            dataloaders = torch.utils.data.DataLoader(
                image_datasets,
                batch_size=opt.batchsize,
                shuffle=False,
                num_workers=8
            )

            candidate_paths = image_datasets.candidates['index']
            candidate_films = np.repeat(movie, candidate_paths.shape[0]).astype(object)
            cast_paths      = image_datasets.casts['index']
            cast_films      = np.repeat(movie, cast_paths.shape[0]).astype(object)

            # Extract feature
            with torch.no_grad():
                features = pcb_extractor.extract_feature(model, dataloaders)

                # candidates first
                num_candidates    = dataloaders.dataset.candidates.shape[0]                
                candidate_feature = features[:num_candidates]
                cast_feature      = features[num_candidates:]

            candidate_feature = candidate_feature.numpy()
            candidate_names   = np.asarray([os.path.basename(name).split('.')[0] for name in candidate_paths.tolist()])
            candidate_films   = np.asarray([name for name in candidate_films.tolist()])
            cast_feature      = cast_feature.numpy()
            cast_names        = np.asarray([os.path.basename(name).split('.')[0] for name in cast_paths.tolist()])
            cast_films        = np.asarray([name for name in cast_films.tolist()])

            # ----------------- #
            # Save to .mat file #
            # ----------------- #
            result = {
                'candidate_features': candidate_feature, 
                'candidate_names': candidate_names,
                'candidate_films': candidate_films,
                'cast_features': cast_feature, 
                'cast_names': cast_names,
                'cast_films': cast_films, 
            }

            mat_path = os.path.join('./features', movie + '_result.mat')
            print("Features saved to {}".format(mat_path))
            scipy.io.savemat(mat_path, result)

        return

    if opt.features:
        results_gpu, results_rerank = [], []

        for movie in tqdm(sorted(os.listdir(opt.testset))):
            feature_path = os.path.join('./features', movie + '_result.mat')
            result = scipy.io.loadmat(feature_path)

            cast_feature = result['cast_features']
            cast_names   = result['cast_names']
            cast_films   = result['cast_films']
            candidate_feature = result['candidate_features']
            candidate_names   = result['candidate_names']
            candidate_films   = result['candidate_films']

            # print("Cast_feature.shape {}".format(cast_feature.shape))
            # print("Cast_film.shape:   {}".format(cast_films.shape))
            # print("Cast_name.shape:   {}".format(cast_names.shape))
            # print("Candidate_feature.shape: {}".format(candidate_feature.shape))
            # print("Candidate_name.shape: {}".format(candidate_names.shape))
            # print("Candidate_film.shape: {}".format(candidate_films.shape))

            result = evaluate_gpu.predict_1_movie(cast_feature, cast_names, candidate_feature, candidate_names)   
            results_gpu.extend(result)

            result = evaluate_rerank.predict_1_movie(cast_feature, cast_names, candidate_feature, candidate_names, opt.k1, opt.k2, opt.lambda_value)
            results_rerank.extend(result)

        with open('gpu_' + opt.output, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
            writer.writeheader()
            
            for r in results_gpu:
                writer.writerow(r)

        with open('rerank_' + opt.output, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
            writer.writeheader()

            for r in results_rerank:
                writer.writerow(r)
            
        mAP = final_eval.eval('gpu_' + opt.output, opt.gt)
        print("mAP(with default dot product) {:.2%}: ".format(mAP))

        mAP = final_eval.eval('rerank_' + opt.output, opt.gt)
        print("mAP(with reranking strategic): {:.2%}: ".format(mAP))

        return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    # Device Setting
    parser.add_argument('--gpu_ids', default=[0], nargs='*', type=int, help='gpu_ids: e.g. 0  0 1 2  0 2')
    # Model and dataset Setting
    parser.add_argument('--resume', type=str, help='Directory to the checkpoint')
    parser.add_argument('--testset', default='./IMDb/val', type=str, help='Directory of the validation set')
    parser.add_argument('--output', default='./result.csv', type=str, help='Directory of the result file.')
    parser.add_argument('--gt', default='./IMDb/val_GT.json', type=str, help='Directory of the output path')
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
    parser.add_argument('--features', type=str, help='Directory of the features.mat')
    parser.add_argument('--img_size', default=[448, 448], type=int, nargs='*')
    # I/O Setting
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    # Model Setting
    parser.add_argument('--num_part', default=6, type=int, help='A parameter of PCB network.')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121')
    parser.add_argument('--use_NAS', action='store_true', help='use NAS')
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    # parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--ms', default=[1.0], nargs='*', type=float, help="multiple_scale: e.g. '1' '1 1.1'  '1 1.1 1.2'")
    # Set k-reciprocal Encoding
    parser.add_argument('--k1', default=20, type=int)
    parser.add_argument('--k2', default=6, type=int)
    parser.add_argument('--lambda_value', default=0.3, type=float)

    opt = parser.parse_args()
    
    utils.details(opt)

    os.makedirs(os.path.join('./feature'), exist_ok=True)

    main(opt)
