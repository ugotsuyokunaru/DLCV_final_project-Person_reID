# -*- coding: utf-8 -*-
"""
  FileName     [ inference_csv.py ]
  PackageName  [ final ]
  Synopsis     [ To inference trained model with testing images, output csv file ]

  Example:
  - python3 inference_csv.py --action test --dataroot ./IMDb_resize/ --model ./net_best.pth --out_csv ./result/ --save_feature
  >> 

  - python3 inference_csv.py --action val --dataroot ./IMDb_resize/ --model ./net_best.pth --out_csv ./result/
  >> 

  # New argparser
  - python3 inference_csv.py --model_features ~ --model_classifier ~ --out_csv ./validatio/ --out_dim 2048 --action val 
                               rerank --k1 20 40 --k2 6 10 --lambda_value 0.15

  - python3.7 inference_csv.py --dataroot ./IMDb_resize/ --model_features ./model_face_4096/resnet50_032.pth --model_classifier ./model_face_4096/classifier_032.pth --out_csv ./validation/ --out_dim 4096 --action val rerank --k1 20 40 --k2 6 10 --lambda_value 0.15
  >>
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
from model_res50 import (Classifier, FeatureExtractorFace,
                         FeatureExtractorOrigin)

newline = '' if sys.platform.startswith('win') else '\n'

def test(castloader: DataLoader, candloader: DataLoader, cast_data, cand_data, 
         feature_extractor: nn.Module, classifier: nn.Module, 
         opt, device, feature_dim=1024, k1=20, k2=6, lambda_value=0.3, mute=False) -> list:
    '''
      Inference by trained model, generated inferenced result if needed.

      Return: 
      - mAP if action == 'val'
      - mAP is 0 for action == 'test'
    '''
    print('Start Inferencing {} dataset ... '.format(opt.action))    

    classifier.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    
    # Constant setting
    mAP = 0
    results_cosine = []
    results_rerank = []

    # --------------------------------- # 
    # If ground truth exists            # 
    # --------------------------------- #
    if opt.action == 'val':
        for i, (cast, labels, moviename, cast_names) in enumerate(castloader, 1):
            print("[{:3d}/{:3d}]".format(i, len(castloader)))

            moviename = moviename[0]

            cast = cast.to(device) # cast_size = 1, num_cast + 1, 3, 448, 448
            cast_out = feature_extractor(cast.squeeze(0))
            cast_out = classifier(cast_out)
            cast_out = cast_out.detach().cpu().view(-1, feature_dim)
            cast_names = [x[0] for x in cast_names]
            
            cand_data.set_mov_name_val(moviename)
            cand_data.mv = moviename

            cand_out = torch.tensor([])
            cand_names = []
            for j, (cand, _, cand_name) in enumerate(candloader):
                cand = cand.to(device)  # cand_size = bs, 3, w, c
                
                if feature_extractor is not None:
                    out = feature_extractor(cand)
                    out = classifier(out)
                else:
                    out = classifier(cand)

                out = out.detach().cpu().view(-1, feature_dim)
                cand_out  = torch.cat((cand_out, out), dim=0)
                cand_names.extend(cand_name)   
        
            casts_features = cast_out.to(device)
            candidates_features = cand_out.to(device)

            print('[Testing]', moviename, 'processed ...', cand_out.size()[0])
            
            cast_names = np.asarray(cast_names, dtype=object)
            cand_names = np.asarray(cand_names, dtype=object)

            result = evaluate.cosine_similarity(casts_features, cast_names, candidates_features, cand_names, mute=mute)
            results_cosine.extend(result)
            
            result = evaluate_rerank.predict_1_movie(casts_features, cast_names, candidates_features, cand_names, 
                                        k1=k1, k2=k2, lambda_value=lambda_value)
            results_rerank.extend(result)

    # --------------------------------- # 
    # If ground truth doesn't exists    # 
    # --------------------------------- #
    if opt.action == 'test':
        for i, (cast, moviename, cast_names) in enumerate(castloader, 1):
            print("[{:3d}/{:3d}]".format(i, len(castloader)))

            moviename = moviename[0]    # unpacked from batch

            cast_names = [x[0] for x in cast_names]

            if not opt.load_feature:
                print("generating {}'s cast features".format(moviename))

                cast = cast.to(device)          # cast.size[1, num_cast, 3, 224, 224]
                cast_out = feature_extractor(cast.squeeze(0))
                cast_out = classifier(cast_out)
                cast_out = cast_out.detach().cpu().view(-1, feature_dim)

                casts_features = cast_out.to(device)

                print("generating {}'s candidate features".format(moviename))
                
                cand_data.set_mov_name_test(moviename)
                cand_out = torch.tensor([])
                cand_names = []
                for j, (cand, cand_name) in enumerate(candloader):
                    cand_names.extend(cand_name)
                    
                    cand = cand.to(device)
                    out = feature_extractor(cand)
                    out = classifier(out)
                    
                    out = out.detach().cpu().view(-1, feature_dim)
                    
                    cand_out = torch.cat((cand_out, out), dim=0)

                candidates_features = cand_out.to(device)

                cast_names = np.asarray(cast_names, dtype=object)
                cand_names = np.asarray(cand_names, dtype=object)

                # Save cast features 
                if opt.save_feature:
                    feature_path = './inference/test/{}/cast/'.format(moviename)
                    os.makedirs(feature_path, exist_ok=True)
                    np.save(os.path.join(feature_path, "features.npy"), casts_features)
                    np.save(os.path.join(feature_path, "names.npy"), cast_names)

                # Save candidates features
                if opt.save_feature:
                    feature_path = './inference/test/{}/candidates/'.format(moviename)
                    os.makedirs(feature_path, exist_ok=True)
                    np.save(os.path.join(feature_path, "features.npy"), candidates_features)
                    np.save(os.path.join(feature_path, "names.npy"), cand_names)

                print('imgs_num({}) / file_names({})'.format(cand_out.size()[0], len(cand_names)))
            
            else:   # if load_feature:
                print("loading {}'s cast features".format(moviename))
                feature_path = './inference/test/{}/cast/features.npy'.format(moviename)
                names_path = './inference/test/{}/cast/names.npy'.format(moviename)
                casts_features = np.load(feature_path)
                cast_names     = np.load(names_path)

                print("loading {}'s candidate features".format(moviename))
                feature_path = './inference/test/{}/candidates/features.npy'.format(movienmae)
                names_path = './inference/test/{}/candidates/names.npy'.format(moviename)
                candidates_features = np.load(feature_path)
                cand_names          = np.load(names_path)

                print('file_names({})'.format(len(cand_names)))

            print('[Testing] {} processing predict_ranking ... \n'.format(moviename))
            
            # predict_ranking
            result = evaluate.cosine_similarity(casts_features, cast_names, candidates_features, cand_names, mute=mute)
            results_cosine.extend(result)
            result = evaluate_rerank.predict_1_movie(casts_features, cast_names, candidates_features, cand_names,
                                        k1=k1, k2=k2, lambda_value=lambda_value)
            results_rerank.extend(result)
    
    mAPs = []
    for submission, results in (('cosine.csv', results_cosine), ('rerank.csv', results_rerank)):
        os.makedirs(opt.out_folder, exist_ok=True)
        path = os.path.join(opt.out_folder, submission)
    
        with open(path, 'w', newline=newline) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print('Testing output "{}" writed. \n'.format(path))

        if opt.action == 'val':
            mAP, AP_dict = final_eval.eval(path, os.path.join(opt.dataroot, "val_GT.json"))
            
            if not mute:
                for key, val in AP_dict.items():
                    record = 'AP({}): {:.2%}'.format(key, val)
                    print(record)    
            
            print('[ mAP = {:.2%} ]\n'.format(mAP))

        mAPs.append(mAP)

    return mAPs

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda")

    folder_name = opt.action

    # ------------------------- # 
    # Dataset initialize        # 
    # ------------------------- #
    if opt.load_feature:
        transform = transforms.ToTensor()

    if not opt.load_feature:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Candidate Dataset and DataLOader
    test_data = CandDataset(
        data_path=os.path.join(opt.dataroot, folder_name),
        drop_others=False,
        transform=transform,
        action=opt.action
    )
        
    test_cast_data = CastDataset(
        data_path=os.path.join(opt.dataroot, folder_name),
        drop_others=False,
        transform=transform,
        action=opt.action
    )

    test_cand = DataLoader(test_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)
    test_cast = DataLoader(test_cast_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    
    print(opt)

    if not opt.command: # Default validation / inference 
        # ------------------------- # 
        # Model initialize          # 
        # ------------------------- #
        feature_extractor = FeatureExtractorFace()# .to(device)
        classifier = Classifier(fc_in_features=2048, fc_out=opt.out_dim)# .to(device)
        
        if opt.model_features:
            print("Parameter read: {}".format(opt.model_features))
            feature_extractor = utils.load_network(feature_extractor, opt.model_features).to(device)

        if opt.model_classifier:
            print("Parameter read: {}".format(opt.model_classifier))
            classifier = utils.load_network(classifier, opt.model_classifier).to(device).to(device)

        # ------------------------- # 
        # Execute Test Function     # 
        # ------------------------- #
        with torch.no_grad():
            test(test_cast, test_cand, test_cast_data, test_data, 
                feature_extractor, classifier, opt, device, 
                k1=40, k2=6, lambda_value=0.1, feature_dim=opt.out_dim, mute=False)
        
        return
    
    if opt.command == 'rerank':
        max_length = max([len(opt.k1), len(opt.k2), len(opt.lambda_value)])
        
        configs = itertools.product(opt.k1, opt.k2, opt.lambda_value)
        feature_extractor = FeatureExtractorFace().to(device)
        classifier = Classifier(fc_in_features=2048, fc_out=opt.out_dim).to(device)
        
        if opt.model_features:
            print("Parameter read: {}".format(opt.model_features))
            feature_extractor = utils.load_network(feature_extractor.cpu(), opt.model_features).to(device)

        if opt.model_classifier:
            print("Parameter read: {}".format(opt.model_classifier))
            classifier = utils.load_network(classifier.cpu(), opt.model_classifier).to(device).to(device)

        # ------------------------- # 
        # Execute Test Function     # 
        # ------------------------- #
        history = []

        for k1, k2, value in configs:
            print("[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}]".format(k1, k2, value))
        
            with torch.no_grad():
                mAPs = test(test_cast, test_cand, test_cast_data, test_data, 
                    feature_extractor, classifier, opt, device, 
                    k1=k1, k2=k2, lambda_value=value, feature_dim=opt.out_dim, mute=True)
                
            history.append(mAPs)

        for (k1, k2, value), history in zip(configs, history):
            print("[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}] [Cosine] mAP: {:.4f}".format(k1, k2, value, history[0]))
            print("[k1: {:3d}, k2: {:3d}, lambda_value: {:.4f}] [Rerank] mAP: {:.4f}".format(k1, k2, value, history[1]))

        return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='inference_csv.py', description='Testing')
    # Dataset setting
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize in testing (one movie folder each time) ')
    # I/O Setting (important !!!)
    parser.add_argument('--model_features', help='model checkpoint path to extract features')    # ./model_face/net_best.pth
    parser.add_argument('--model_classifier', help='model checkpoint path to classifier')
    parser.add_argument('--dataroot', default='./IMDb_resize/', type=str, help='Directory of dataroot')
    parser.add_argument('--action', default='test', type=str, help='action type (test / val)')
    parser.add_argument('--out_dim', default=1024, type=int, help='to set the output dimensions of FC Layer')
    parser.add_argument('--gt', type=str, help='if gt_file is exists, measure the mAP.')
    parser.add_argument('--out_folder',  default='./inference/', help='output csv folder name')
    parser.add_argument('--save_feature', action='store_true', help='save new np features when processing')
    parser.add_argument('--load_feature', action='store_true', help='load old np features when processing')
    # Device Setting
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')
    
    # Rerank Setting
    subparser = parser.add_subparsers(dest='command', help='Advanced option')
    rerank_parser = subparser.add_parser('rerank', help='scanning reranking function')
    rerank_parser.add_argument('--k1', default=[20], nargs='*', type=int)
    rerank_parser.add_argument('--k2', default=[6], nargs='*', type=int)
    rerank_parser.add_argument('--lambda_value', default=[0.3], nargs='*', type=float)

    opt = parser.parse_args()
    
    # Prints the setting
    utils.details(opt)

    # Make directories
    os.makedirs('./inference/', exist_ok=True)
    os.makedirs('./inference/val/', exist_ok=True)
    os.makedirs('./inference/test/', exist_ok=True)

    # Check files exists
    if not os.path.exists(opt.dataroot):
        raise IOError("{} is not exists".format(opt.dataroot))
    
    if opt.save_feature and opt.load_feature:
        raise ValueError('Cannot save and load features simultanesouly, please choose one of them')
    
    # if not os.path.exists(opt.gt):
    #     pass
    #     raise IOError("{} is not exists".format(opt.gt))

    # Execute the main function
    main(opt)
