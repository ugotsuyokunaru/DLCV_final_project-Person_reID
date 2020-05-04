"""
  FileName     [ evaluate_rerank.py ]
  PackageName  [ final ]
  Synopsis     [ To evaluate the model performance ]
"""

import argparse
import csv
import os
import time

import numpy as np
from numpy import linalg as LA
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

import final_eval
import utils
from evaluate import normalize_ndarray
from re_ranking import re_ranking


# This evaluate function is different with other evaluate functions
def evaluate(score, ql, qc, gl, gc):
    """
      Params:
      - score
      - ql: query_label = result['query_label'][0]
      - qc: query_cam = result['query_cam'][0]
      - gl: gallery_cam = result['gallery_cam'][0]
      - gc: gallery_label = result['gallery_label'][0]
    
      Return:
      - CMC_tmp
    """
    index = np.argsort(score)  #from small to large
    # index = index[::-1]
    
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    return good_index, junk_index

def main(opt):
    # ------------ #
    # Read results #
    # ------------ #
    result = scipy.io.loadmat(opt.features)

    cast_feature = result['cast_features']
    cast_name    = result['cast_names'].reshape(-1)
    cast_film    = result['cast_films'].reshape(-1)

    candidate_feature = result['candidate_features']
    candidate_name    = result['candidate_names'].reshape(-1)
    candidate_film    = result['candidate_films'].reshape(-1)

    print("Cast_feature.shape {}".format(cast_feature.shape))
    print("Cast_film.shape:   {}".format(cast_film.shape))
    print("Cast_name.shape:   {}".format(cast_name.shape))
    print("Candidate_feature.shape: {}".format(candidate_feature.shape))
    print("Candidate_name.shape: {}".format(candidate_name.shape))
    print("Candidate_film.shape: {}".format(candidate_film.shape))

    run(cast_feature, cast_name, cast_film, candidate_feature, candidate_name, candidate_film, opt.gt, opt.output)

def run(cast_feature, cast_name, cast_film, candidate_feature, candidate_name, candidate_film, gt, output, k1=20, k2=6, lambda_value=0.3):
    cast_name = cast_name[:-1]
    cast_film = cast_film[:-1]

    films = np.unique(cast_film)
    # ------------ #
    # Reranking    #
    # ------------ #

    result = []
    for i in range(films.shape[0]):
        film = films[i]
        
        mask_candidate = (candidate_film == film).astype(bool)
        mask_cast      = (cast_film == film).astype(bool)

        q = cast_feature[mask_cast]
        q_name = cast_name[mask_cast]

        g = candidate_feature[mask_candidate]
        g_name = candidate_name[mask_candidate]

        # print('calculate initial distance')
        q_g_distance = np.dot(q, np.transpose(g))
        q_q_distance = np.dot(q, np.transpose(q))
        g_g_distance = np.dot(g, np.transpose(g))
        # print(q_g_distance.shape, q_q_distance.shape, g_g_distance.shape)

        final_distance = re_ranking(q_g_distance, q_q_distance, g_g_distance, k1=k1, k2=k2, lambda_value=lambda_value)
        
        for j in range(final_distance.shape[0]):
            distance = final_distance[j]
            index = np.argsort(distance)
            
            cast_id = q_name[j]
            candidates = g_name[index]

            result.append({
                'Id': cast_id, 
                'Rank': ' '.join(candidates)
            })

    # Open with binary on windows
    with open(output, 'w', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])

        writer.writeheader()
        for r in result:
            writer.writerow(r)

    mAP = final_eval.eval(output, gt)
        
    return mAP

def predict_1_movie(cast_feature: torch.Tensor, cast_name, candidate_feature, candidate_name, k1=20, k2=6, lambda_value=0.3) -> list:
    """
      Input
      - cast_feature:       numpy array[n, feature_dim] (float)
      - cast_name:          numpy array[n, ] (str)
      - candidate_feature:  numpy array[m, feature_dim] (float)
      - candidate_name:     numpy array[m, ] (str)
    """
    # cast_feature      = cast_feature.reshape(cast_feaure.shape[0], -1)
    # candidate_feature = candidate_feature.reshape(candidate_feature.shape[0], -1)
    
    cast_feature      = torch.nn.functional.normalize(cast_feature, dim=1)
    candidate_feature = torch.nn.functional.normalize(candidate_feature, dim=1)

    q_g_distance = torch.mm(cast_feature, candidate_feature.transpose(0, 1)).cpu().numpy()
    q_q_distance = torch.mm(cast_feature, cast_feature.transpose(0, 1)).cpu().numpy()
    g_g_distance = torch.mm(candidate_feature, candidate_feature.transpose(0, 1)).cpu().numpy()
    
    # Re_ranking() using L2 Distance as the result, smaller distance mean 'closer' with each other
    final_distance = re_ranking(q_g_distance, q_q_distance, g_g_distance, k1=k1, k2=k2, lambda_value=lambda_value)

    result = []
    for j in range(final_distance.shape[0]):
        distance = final_distance[j]
        index = np.argsort(distance)
        
        cast_id = cast_name[j]
        candidates = candidate_name[index]
        
        result.append({
            'Id': cast_id, 
            'Rank': ' '.join(candidates)
        })
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='evaluate_rerank.py', description='Reranking function')
    parser.add_argument('--gt', type=str, help='directory of the gt.json')
    parser.add_argument('--output', type=str, help='directory of the output.csv')
    opt = parser.parse_args()

    utils.details(opt)

    main(opt)
