"""
  FileName     [ evaluate_gpu.py ]
  PackageName  [ final ]
  Synopsis     [ To generate the query_list by using dot product to measure 
                 the distances between cast and candidates. ]

  Usage:
  - python3 evaluate_gpu.py --name PCB
"""
import csv
import os
import argparse

import numpy as np
import scipy.io
import torch

def evaluate(qf, ql, qc, gf, gl, gc, default_juke_label='others'):
    # qf contains 1 image feature       # query feature (tensor)
    # print("Qf.shape: ", qf.shape)     # query label   (numpy array)
    # print("GF.shape: ", gf.shape)     # gallery feature (tensor)
                                        # query camera (movie 概念，但一次只丟一部，沒差)
    query = qf.view(-1,1)
    # print(query.shape)
    
    # ------------------------------------
    # Dot product
    #   score[i] = np.dot(gf[i], query)
    # ------------------------------------
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]        # inverse it

    return index

    # if (ql is None) or (qc is None) or (gl is None) or (gc is None):
    #     return index

    # # ----------------------------------
    # # Calculating the mAP and CMC here
    # # ----------------------------------
    # # good index
    # query_index  = np.argwhere(gl==ql)
    # camera_index = np.argwhere(gc==qc)

    # # ----------------------------------------------------------------------
    # # np.intersect1d(arr1, arr2)
    # #   Treat the np.array() as a set, find the intersection between 2 sets
    # # 
    # # np.setdiff1d(arr1, arr2)
    # #   Set subtraction, return the elements in arr1 but not in arr2
    # # ----------------------------------------------------------------------

    # good_index  = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gl==-1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index  = np.append(junk_index2, junk_index1)
    
    # # Cumulative match curve = CMC
    # CMC_tmp = compute_mAP(index, good_index, junk_index)
    # return index, CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate_GPU')
    parser.add_argument('--features', type=str, help='directory of the features of cast and candidates.')
    parser.add_argument('--multi', action='store_true', help='if true, ranking multi-cast at the same time')
    parser.add_argument('--predict', action='store_true', help='if true, label is un-availables.')
    parser.add_argument('--output', type=str)


    opt = parser.parse_args()

    result = scipy.io.loadmat(opt.features)
    
    # print(result['cast_paths'].shape)
    # print(result['cast_paths'])

    cast_feature = torch.FloatTensor(result['cast_features'])
    cast_name = result['cast_names'].reshape(-1)
    cast_film = result['cast_films'].reshape(-1)
    print(cast_name)
    print(cast_film)
    candidate_feature = torch.FloatTensor(result['candidate_features'])
    candidate_name = result['candidate_names'].reshape(-1)
    candidate_film = result['candidate_films'].reshape(-1)
    print(candidate_name)
    print(candidate_film)

    if not opt.multi:
        cast_feature = cast_feature.cuda()
        candidate_feature = candidate_feature.cuda()

        print("cast_feature.shape: ", cast_feature.shape)
        print("cast_film.shape:    ", cast_film.shape)
        print("cast_name.shape:    ", cast_name.shape)
        # print(cast_path)

        CMC = torch.IntTensor(len(candidate_film)).zero_()
        ap = 0.0

        result = []
        for i in range(cast_feature.shape[0]):
            print("[{}/{}]".format(i, cast_feature.shape[0]))
            
            mask = torch.from_numpy((candidate_film == cast_film[i]).astype(np.uint8)).byte()
            # print("select_film: ", cast_film[i])
            # print("select_candidate.shape: ", candidate_film[mask].shape)
            # print(mask.dtype)
            
            index, (ap_tmp, CMC_tmp) = evaluate(
                cast_feature[i], cast_name[i], cast_film[i], 
                candidate_feature[mask], candidate_name[mask], candidate_film[mask]
            )
            # print("Index.shape: ", index.shape)
            # print(cast_path[i].tolist()[0].split('/')[-1].split('.')[0])
            # print(type(candidate_path[index].tolist()))
            names = candidate_name[index].tolist()
            cast_id = os.path.basename(cast_name.tolist()[i]).split('.')
            result.append({'Id': cast_id, 'Rank': ' '.join(names)})

            if CMC_tmp[0] == -1 :continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        with open('result.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])

            writer.writeheader()
            for r in result:
                writer.writerow(r)


        CMC = CMC.float()
        CMC = CMC / cast_feature.shape[0] #average CMC
        print('\nRank@1:%f\nRank@5:%f\nRank@10:%f\nmAP:%f'%(CMC[0], CMC[4], CMC[9], ap / len(cast_name)))

    if opt.multi:
        # Not tested.
        raise NotImplementedError

        # m_result = scipy.io.loadmat('./mat/multi_query_{}.mat'.format(opt.name))
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()
        # multiple-query
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
    
        for i in range(len(query_label)):
            mquery_index1 = np.argwhere(mquery_label==query_label[i])
            mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
            mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
            mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
            ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])
        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
