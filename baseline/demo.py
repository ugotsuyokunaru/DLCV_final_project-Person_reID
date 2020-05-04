"""
  FileName     [ demo.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Generate 1 images sequence to demo the Person_reID effect/ ]

  Dataset:
  - Market1501

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.

  Pretrain network:
  - PCB:
  - DenseNet:
  - NAS:
  - ResNet: 

  Usage:
  - python3 demo.py --name ft_net_dense
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torchvision import datasets

# matplotlib.use('Agg')


#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='./Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_net_dense', type=str, help='load model path')

opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
def imshow(path, title=None, pause_time=0.5):
    """ Imshow for Tensor. """
    im = plt.imread(path)
    plt.imshow(im)

    if title is not None:
        plt.title(title)
    
    plt.pause(pause_time)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('./mat/pytorch_result_{}.mat'.format(opts.name))
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('./mat/multi_query_{}.mat'.format(opts.name))

if multi:
    m_result = scipy.io.loadmat('./mat/multi_query_{}.mat'.format(opts.name))
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    """
      Params:
      - qf: Query
      - ql:
      - qc: Camera_index
      - gf:
      - gl:
      - gc:
    """
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    
    return index

i = opts.query_index
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

#-------------------------------
# Visualize the rank result
# ------------------------------
if __name__ == "__main__":
    query_path, _ = image_datasets['query'].imgs[i]
    query_label = query_label[i]
    print(query_path)
    print('Top 10 images are as follow:')
    
    try: # Visualize Ranking Result 
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'].imgs[index[i]]
            label = gallery_label[index[i]]
            imshow(img_path)
            if label == query_label:
                ax.set_title('%d'%(i+1), color='green')
            else:
                ax.set_title('%d'%(i+1), color='red')
            print(img_path)
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    os.makedirs('./demo_output/', mode=0o777, exist_ok=True)
    fig.savefig("./demo_output/show.png")
    print("\nresult saved to path : ./demo_output/show.png\n")
