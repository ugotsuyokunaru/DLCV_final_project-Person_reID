"""
  FileName     [ prepare.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Put the images with the same id in one folder. ]

  For Market1501 dataset:
    The images with different IDs are mixed in one folder, we needed to seperated it first.

  Notes:
  - You need to modify the value of variable 'download_path' to run this script.
"""

import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = './Market-1501-v15.09.15'

def make_ids(source_path, target_path):
    """
      Params:
      - source_path
      - target_path

      Return: None
    """
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    for _, _, files in os.walk(source_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg': continue
            
            ID = name.split('_')[0]
            src_name   = os.path.join(source_path, name)
            dst_folder = os.path.join(target_path, ID)

            if not os.path.isdir(dst_folder):
                os.mkdir(dst_folder)
            
            copyfile(src_name, os.path.join(dst_folder, name))

    return

if __name__ == "__main__":
    if not os.path.isdir(download_path):
        print("Download_path {} doesn't exist.".format(download_path))

    save_path = os.path.join(download_path, 'pytorch')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # ----------------------------------------------
    # 1. single-query
    # 2. Multi-query (for dukemtmc-reid, we do not need multi-query)
    # 3. Gallery
    # 4. Train_all
    # ----------------------------------------------
    source_paths = [
        os.path.join(download_path, 'query'),
        os.path.join(download_path, 'gt_bbox'),
        os.path.join(download_path, 'bounding_box_test'),
        os.path.join(download_path, 'bounding_box_train'),
    ]
    target_paths = [
        os.path.join(save_path, 'query'),
        os.path.join(save_path, 'multi-query'),
        os.path.join(save_path, 'gallery'),
        os.path.join(save_path, 'train_all'),
    ]

    for src_path, tgt_path in zip(source_paths, target_paths):
        # Make 4 sub-datasets with default methods.
        print('{}\n{}'.format(src_path, tgt_path))
        if os.path.isdir(src_path):
            make_ids(src_path, tgt_path)

    # ---------------------------------------
    # train_val
    # ---------------------------------------
    # first image is used as val image
    train_path      = os.path.join(download_path, 'bounding_box_train')
    train_save_path = os.path.join(save_path, 'train')
    val_save_path   = os.path.join(save_path, 'val')
    
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.isdir(val_save_path):
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg': continue
            
            ID  = name.split('_')[0]
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + ID
                os.mkdir(dst_path)
            
            copyfile(src_path, dst_path + '/' + name)
