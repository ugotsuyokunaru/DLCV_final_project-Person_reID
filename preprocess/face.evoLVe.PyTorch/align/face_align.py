import argparse
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from align_trans import get_reference_facial_points, warp_and_crop_face
from detector import detect_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default="./IMDb/train", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default="./IMDb_preprocess/train", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default=112, type = int)
    args = parser.parse_args()
    
    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    # '.DS_Store' file Only exists in MacOS
    # cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")    
    # os.chdir(cwd)

    if not os.path.exists(source_root):
        raise IOError

    if not os.path.isdir(dest_root):
        os.makedirs(dest_root, exist_ok=True)

    movies = sorted(os.listdir(source_root))
    for movie in tqdm(movies):
        source_root = os.path.join(args.source_root, movie)

        if not os.path.isdir(os.path.join(dest_root, movie)):
            os.mkdir(os.path.join(dest_root, movie))

        for subfolder in os.listdir(source_root):
            if not os.path.isdir(os.path.join(source_root, subfolder)):
                copyfile(os.path.join(source_root, subfolder), os.path.join(dest_root, movie, subfolder))
                print('File {} copied'.format(os.path.join(dest_root, movie, subfolder)))
                continue
            
            if not os.path.exists(os.path.join(dest_root, movie, subfolder)):
                os.mkdir(os.path.join(dest_root, movie, subfolder))
            
            crop_count = 0
            for image_name in tqdm(sorted(os.listdir(os.path.join(source_root, subfolder)))):
                # print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
                img = Image.open(os.path.join(source_root, subfolder, image_name))

                try:
                    _, landmarks = detect_faces(img, model_paths=[
                        './preprocess/face.evoLVe.PyTorch/align/pnet.npy', 
                        './preprocess/face.evoLVe.PyTorch/align/rnet.npy', 
                        './preprocess/face.evoLVe.PyTorch/align/onet.npy']
                    )                
                except Exception:
                    # print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                    img = img.resize(size=((args.crop_size, args.crop_size)), resample=Image.BICUBIC)
                    img.save(os.path.join(dest_root, movie, subfolder, image_name))

                    continue
                
                # If the landmarks cannot be detected, the img will be resized only
                if len(landmarks) == 0:
                    # print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                    img = img.resize(size=((args.crop_size, args.crop_size)), resample=Image.BICUBIC)
                    img.save(os.path.join(dest_root, movie, subfolder, image_name))
                    
                    continue
                
                # Crop the images by the first landmarks information (adjustable)
                if len(landmarks) > 0:
                    # print("{} have {} landmarks.".format(os.path.join(source_root, subfolder, image_name), len(landmarks)))
                    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
                    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                    img_warped = Image.fromarray(warped_face)

                    if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                        image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
                    
                    img_warped.save(os.path.join(dest_root, movie, subfolder, image_name))

                    crop_count += 1

            print("{} crops {:4d} / {:4d} images.".format(os.path.join(os.path.basename(dest_root), movie, subfolder), crop_count, len(os.listdir(os.path.join(source_root, subfolder)))))
