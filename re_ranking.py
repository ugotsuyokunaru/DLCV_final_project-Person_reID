#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]

    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3) -> np.ndarray:
    """
      q_g_distance: 
        np.dot(q, g.transpose())
      q_q_distance: 
        np.dot(q, q.transpose())
      g_g_distance:
        np.dot(g, g.transpose())
    """
    # --------------------------------- #
    # Dimension:                        #
    #   m: the number of galleries      #
    #   n: the number of queries        #
    #   original_dist: (m + n), (m + n) #
    # --------------------------------- #
    
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
        np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0
    )
    
    # change the cosine similarity metric to euclidean similarity metric
    original_dist = 2. - 2 * original_dist   
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    
    # ------------------------------------------------------------------- #
    # np.argpartition():                                                  # 
    #   Put the k-th smallest element at the k-th position,               #
    #   The left of the k-th position are the elements smaller than k-th  #
    #   The right of the k-th position are the elements largher than k-th #
    # ------------------------------------------------------------------- #

    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1 (the 0-th smallest is self)
    initial_rank = np.argpartition(original_dist, range(1, k1+1))

    query_num = q_g_dist.shape[0]       # n
    all_num = original_dist.shape[0]    # m + n

    for i in range(all_num):
        # R(p, k)
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        # R*(p, k)
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in range(len(k_reciprocal_index)):
            # q
            candidate = k_reciprocal_index[j]
            # R(q, 0.5k)
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1/2)))
            # R*(p, k) <-- R(p, k) intersection R(q, 0.5k)
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2./3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        # Tide up R*(p, k)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank
    
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2.-temp_min)

    
    final_dist = jaccard_dist * (1-lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist

    final_dist = final_dist[:query_num,query_num:]
    return final_dist
