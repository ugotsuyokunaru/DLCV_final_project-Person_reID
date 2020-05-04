# -*- coding: utf-8 -*-
"""
  FileName     [ inference_csv.py ]
  PackageName  [ final ]
  Synopsis     [ To inference trained model with images generate features and save to .npy file ]

  Example:
    python3.7 preprocess_features.py --dataroot ./IMDb_resize/
"""
import argparse
import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import evaluate_rerank
import final_eval
import utils
from imdb import CastDataset, CandDataset
from model_res50 import FeatureExtractorFace, FeatureExtractorOrigin, Classifier

def extractor_features(castloader, candloader, cast_data, cand_data, Feature_extractor, opt, device, folder_name, model_name):
    '''
      Inference by trained model, extracted features and save as .npy file.

      Params:
      - castloader
      - candloader
      - cast_data: the name list of casts
      - cand_data: the name list of candidates

      Return: None
    '''

    print('Start Extracting features of {}ing dataset with model({})... '.format(folder_name, model_name))
    # os.makedirs('./feature_np/', exist_ok=True)
    os.makedirs('./feature_np/{}/{}/'.format(model_name, folder_name), exist_ok=True)
    
    Feature_extractor.eval()
    with torch.no_grad():
        for i, (cast, labels, mov, cast_file_name_list) in enumerate(castloader):
            mov = mov[0]    # unpacked from batch
            cast_file_name_list = [x[0] for x in cast_file_name_list]   # # [('tt0121765_nm0000204',), ('tt0121765_nm0000168',), ...] to ['tt0121765_nm0000204', 'tt0121765_nm0000168', ...]
            cast_labels = np.array(labels[0])   # tensor([[1,2,3,4,5,...]]) to np.arrya([1,2,3,4,5,..])

            # 1. Generate cast features
            print("generating {}'s cast features".format(mov))
            cast = cast.to(device)          # cast.size[1, num_cast, 3, 224, 224]
            cast_out = Feature_extractor(cast.squeeze(0))
            cast_out = cast_out.detach().cpu().view(-1, 2048)
            casts_features = cast_out.numpy()
            # Save cast features 
            feature_path = './feature_np/{}/{}/{}/cast/'.format(model_name, folder_name, mov)
            os.makedirs(feature_path, exist_ok=True)
            np.save(os.path.join(feature_path, "features.npy"), casts_features)
            np.save(os.path.join(feature_path, "names.npy"), cast_file_name_list)
            np.save(os.path.join(feature_path, "labels.npy"), cast_labels)

            print("cast_file_name_list :", cast_file_name_list)
            print('Saved features to {}'.format(feature_path))
            print('imgs_num({}) / file_names({})\n'.format(cast_out.size()[0], len(cast_file_name_list)))

            # 2. Generate candidate features
            print("generating {}'s candidate features".format(mov))
            cand_data.set_mov_name_save(mov)
            cand_out = torch.tensor([])
            cand_file_name_list = []
            label_list = []
            for j, (cand, label_mapped, cand_file_name_tuple) in enumerate(candloader):
                label_list.extend(list(label_mapped.numpy()))
                cand_file_name_list.extend(list(cand_file_name_tuple))
                cand = cand.to(device)
                out = Feature_extractor(cand)
                out = out.detach().cpu().view(-1, 2048)
                cand_out = torch.cat((cand_out, out), dim=0)
            candidates_features = cand_out.numpy()
            # Save candidates features
            feature_path = './feature_np/{}/{}/{}/candidates/'.format(model_name, folder_name, mov)
            os.makedirs(feature_path, exist_ok=True)
            np.save(os.path.join(feature_path, "features.npy"), candidates_features)
            np.save(os.path.join(feature_path, "names.npy"), cand_file_name_list)
            np.save(os.path.join(feature_path, "labels.npy"), label_list)

            print('Saved features to {}'.format(feature_path))
            print('imgs_num({}) / file_names({})\n'.format(cand_out.size()[0], len(cand_file_name_list)))
    print('Extracted all features of {} with model {}.\n'.format(folder_name, model_name))

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda:0")

    transform1 = transforms.Compose([
                        # transforms.Resize((224,224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                             ])
    

    # get fixed model
    for model_name in ['origin', 'face']:
        if model_name == 'origin':
            Feature_extractor = FeatureExtractorOrigin().to(device)
        elif model_name == 'face':
            Feature_extractor = FeatureExtractorFace().to(device)

        # initialize datasets
        for folder_name in ['val', 'train']:
            test_data = CandDataset(
                data_path=os.path.join(opt.dataroot, folder_name), transform=transform1, action='save')
                                        
            test_cand = DataLoader(
                test_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)
            
            test_cast_data = CastDataset(
                data_path=os.path.join(opt.dataroot, folder_name), transform=transform1, action='save')

            test_cast = DataLoader(
                test_cast_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        
            # extract features (total 4 times)
            extractor_features(test_cast, test_cand, test_cast_data, test_data, Feature_extractor, opt, device, folder_name, model_name)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Preprocess')

    # Dataset setting
    parser.add_argument('--dataroot', default='/media/disk1/EdwardLee/dataset/IMDb_resize/', type=str, help='Directory of dataroot')
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize in testing (one movie folder each time) ')

    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')

    opt = parser.parse_args()

    # Check files here
    if not os.path.exists(opt.dataroot):
        raise IOError("{} is not exists".format(opt.dataroot))
    
    utils.details(opt)
    main(opt)