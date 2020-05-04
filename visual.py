"""
  FileName     [ visual.py ]
  PackageName  [ final ]
  Synopsis     [  ]

  Usage:
    python3 visual.py --csv_file <dir+filename.csv>  --cand_num <num> --cast_name <cast_name>
    python3 visual.py --csv_file ./IMDb/sample_submission.csv  --cand_num 5 --cast_name tt1840309_nm0000171 --test_dir ./IMDb_resize/test/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import utils

def main(opts):
    #---------------------------------------------- #
    # Visualize the ranked result from csv file     #
    # --------------------------------------------- #
    csv_result = pd.read_csv(opts.csv_file)
    cast_name = opts.cast_name
    movie_name = cast_name.split('_')[0]
    
    cast_dir = os.path.join(opts.test_dir, movie_name, 'cast', cast_name + '.jpg')
    # cast_dir = os.path.join(opts.test_dir + movie_name + '/cast/' + cast_name + '.jpg')
    cast_img = Image.open(cast_dir)
    cand_dir = os.path.join(opts.test_dir, movie_name, 'candidates')
    # cand_dir = os.path.join(opts.test_dir+movie_name+'/candidates/')
  
    cast_img = np.array(cast_img)
    cand_num = int(opts.cand_num)

    target = csv_result[csv_result['Id'] == cast_name]
    cands = target['Rank'].tolist()[0].split(' ')

    plt.figure(figsize=(20, 10))
    plt.subplot(1, cand_num+1, 1)

    plt.axis('off')
    plt.imshow(cast_img)
        
    for i in range(cand_num):
        cands_img = Image.open(os.path.join(cand_dir, cands[i] + '.jpg'))
        plt.subplot(1, cand_num+1, i+2)
        plt.imshow(cands_img)
        plt.axis('off')
    
    plt.savefig("demo.jpg")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cast_name', default='tt0053221_nm0000078', type=str)
    parser.add_argument('--csv_file', default='./rerank.csv',type=str, help="If 'csv_file' contains value, read csv_file.csv")
    parser.add_argument('--test_dir', default='./IMDb/test', type=str, help='./test_data')
    parser.add_argument('--cand_num', default=5, help='numbers of demo')

    parser.add_argument('--show', action='store_true', help='if true, show the resule with GUI')
    opts = parser.parse_args()

    utils.details(opts)
    main(opts)
