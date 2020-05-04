# -*- coding: utf-8 -*-
"""
  FileName     [ rerank_searching.py ]
  PackageName  [ final ]
  Synopsis     [ ... ]
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import evaluate
import evaluate_rerank
import final_eval
import utils
from imdb import CandDataset, CastDataset
from model_res50 import Classifier, FeatureExtractorFace

newline = '' if sys.platform.startswith('win') else '\n'

def cosine(castloader: DataLoader, candloader: DataLoader, cast_data: CastDataset, cand_data: CandDataset, 
    feature_extractor: nn.Module, classifier: nn.Module, opt, device, feature_dim=2048, mute=True) -> list:

    features = []
    results = []
    
    for i, (cast, _, moviename, cast_names) in enumerate(castloader, 1):
        print("[{:3d}/{:3d}] {}".format(i, len(castloader), moviename))

        moviename = moviename[0]

        cast = cast.to(device) # cast_size = 1, num_cast + 1, 3, 448, 448
        cast_out = feature_extractor(cast.squeeze(0))
        cast_out = classifier(cast_out)
        cast_out = cast_out.detach().cpu().view(-1, feature_dim)
        cast_names = [x[0] for x in cast_names]
        
        cand_data.set_mov_name_val(moviename)
        cand_data.mv = moviename

        # Scanning condidates
        cand_out, cand_names = torch.tensor([]), []
        for j, (cand, _, cand_name) in enumerate(candloader):
            cand = cand.to(device)  # cand_size = bs, 3, w, c
            
            if feature_extractor is not None:
                out = feature_extractor(cand)
                out = classifier(out)
            else:
                out = classifier(cand)

            out = out.detach().cpu().view(-1, feature_dim)
            cand_out = torch.cat((cand_out, out), dim=0)
            cand_names.extend(cand_name)   
    
        casts_features, candidates_features = cast_out.to(device), cand_out.to(device)
        cast_names, cand_names = np.asarray(cast_names, dtype=object), np.asarray(cand_names, dtype=object)
        
        result = evaluate.cosine_similarity(casts_features, cast_names, candidates_features, cand_names, mute=mute)
        results.extend(result)
        features.append((casts_features, cast_names, candidates_features, cand_names))        

    return results, features

def rerank(features, k1=40, k2=6, lambda_value=0.15, mute=True) -> list:
    results = []
    for i, (casts_features, cast_names, candidates_features, cand_names) in enumerate(features, 1):
        # print("[{:3d}/{:3d}]".format(i, len(features)))

        result = evaluate_rerank.predict_1_movie(casts_features, cast_names, candidates_features, cand_names, k1=k1, k2=k2, lambda_value=lambda_value)
        results.extend(result)

    return results

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda")

    # ------------------------- # 
    # Dataset initialize        # 
    # ------------------------- #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Candidate Dataset and DataLoader
    val_data = CandDataset(
        data_path=os.path.join(opt.dataroot, 'val'),
        drop_others=False,
        transform=transform,
        action='val'
    )
        
    val_cast_data = CastDataset(
        data_path=os.path.join(opt.dataroot, 'val'),
        drop_others=False,
        transform=transform,
        action='val'
    )

    val_cand = DataLoader(val_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)
    val_cast = DataLoader(val_cast_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    
    configs = itertools.product(opt.k1, opt.k2, opt.lambda_value)
    
    feature_extractor = FeatureExtractorFace().to(device)
    classifier = Classifier(fc_in_features=2048, fc_out=opt.feature_dim).to(device)
    
    if opt.model_features:
        print("Parameter read: {}".format(opt.model_features))
        feature_extractor = utils.load_network(feature_extractor.cpu(), opt.model_features).to(device)

    if opt.model_classifier:
        print("Parameter read: {}".format(opt.model_classifier))
        classifier = utils.load_network(classifier.cpu(), opt.model_classifier).to(device).to(device)

    feature_extractor.eval()
    classifier.eval()

    # ------------------- # 
    # Cosine Similarity   # 
    # ------------------- #
    results, features = cosine(val_cast, val_cand, val_cast_data, val_data, 
        feature_extractor, classifier, opt, device, feature_dim=opt.feature_dim, mute=True)

    path = os.path.join(opt.out_folder, 'cosine.csv')

    with open(path, 'w', newline=newline) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('Testing output "{}" writed. \n'.format(path))
    mAP, _ = final_eval.eval(path, opt.gt_file)
    print('[ mAP = {:.2%} ]\n'.format(mAP))

    # ------------------- # 
    # Re-Ranking          # 
    # ------------------- #
    mAPs = []
    with open('parameters.txt', 'w') as textfile:
        for k1, k2, value in configs:
            path = os.path.join(opt.out_folder, '_'.join(('rerank', str(k1), str(k2), str(value))) + '.csv')
            # print("[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}]".format(k1, k2, value))
    
            with torch.no_grad():
                results = rerank(features, k1, k2, value, mute=True)
            
            with open(path, 'w', newline=newline) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
                writer.writeheader()
                for r in results:
                    writer.writerow(r)

            # print('Testing output "{}" writed. \n'.format(path))

            mAP, _ = final_eval.eval(path, opt.gt_file)
            mAPs.append(mAP)
            
            message = '[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}] mAP: {:.2%} / {:.2%}'.format(
                        k1, k2, value, mAP, max(mAPs))

            textfile.write(message + '\n')
            print(message)
 
    # ------------------- #
    # Print the results   # 
    # ------------------- #
    for (k1, k2, value), mAP in zip(configs, mAPs):
        print("[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}] [Rerank] mAP: {:.2%}".format(k1, k2, value, mAP))

    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='inference_csv.py', description='Testing')
    # Dataset setting
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize in testing (one movie folder each time) ')
    # I/O Setting (important !!!)
    parser.add_argument('--model_features', default='./models/resnet50.pth', help='model checkpoint path to extract features')    # ./model_face/net_best.pth
    parser.add_argument('--model_classifier', default='./models/classifier.pth', help='model checkpoint path to classifier')
    parser.add_argument('--dataroot', default='./IMDb_Resize', type=str, help='Directory of dataroot')
    parser.add_argument('--feature_dim', default=2048, type=int, help='Output dimensions of FC Layer')
    parser.add_argument('--gt_file', default='./IMDb_Resize/val_GT.json', type=str, help='if gt_file is exists, measure the mAP.')
    parser.add_argument('--out_folder',  default='./inference', help='output csv folder name')
    # Device Setting
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')
    parser.add_argument('--k1', default=[20], nargs='*', type=int)
    parser.add_argument('--k2', default=[6], nargs='*', type=int)
    parser.add_argument('--lambda_value', default=[0.3], nargs='*', type=float)

    opt = parser.parse_args()
    
    # Prints the setting
    utils.details(opt)

    # Make directories
    os.makedirs(opt.out_folder, exist_ok=True)
    os.makedirs(os.path.join(opt.out_folder, 'val'), exist_ok=True)
    
    # Check files exists
    if not os.path.exists(opt.dataroot):
        raise IOError("{} is not exists".format(opt.dataroot))

    if not os.path.exists(opt.gt_file):
        pass
        raise IOError("{} is not exists".format(opt.gt))

    # Execute the main function
    main(opt)
