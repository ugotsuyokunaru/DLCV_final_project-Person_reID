# -*- coding: utf-8 -*-
"""
  FileName     [ tri_loss.py ]
  PackageName  [ final ]
  Synopsis     [ Self defined triplet loss loading methods. ]

  Example:
    python3.7 preprocess_features.py --dataroot ./IMDb_resize/
"""
import torch
import torch.nn as nn

def triplet_loss(inputs, labels: list, cast_num: int, triplet_criterion, norm_criterion=None):
    """
      Self define triplet loss function.

      Params:
      - inputs: Tensor  _(num_cast + num_cand) x 2048
      - labels: List    _(num_cast + num_cand) x 1
      - cast_num: int   _ number of casts

      Description:
      - x_a: The candidates' images
      - x_p: The casts' images
      - x_n: The candidates' images

      Return:
      - loss
    """

    batchsize = inputs.shape[0]
    candidate_num = batchsize - cast_num

    inputs = inputs.reshape(batchsize, -1)

    x_a, x_p = inputs[cast_num:], inputs[:cast_num]

    labels = torch.LongTensor(labels)

    index_a = labels[cast_num:]
    index_n = torch.randint(0, cast_num, size=(candidate_num, ))

    # If index_n is equal to index_a, random again.
    while (index_n == index_a).nonzero().shape[0] > 0:
        n = (index_n == index_a).nonzero().numel()
        index_n[index_n == index_a] = torch.randint(0, cast_num, size=(n, ))

    # Make the P/N Pairs with index_p and index_n
    # print("index_n :", index_n)
    # print("index_p :", index_p)
    # print("x_p.shape ;", x_p.shape)
    # print()
    x_p, x_n = x_p[index_a], x_p[index_n]

    if norm_criterion is not None:
        loss = triplet_criterion(x_a, x_p, x_n) + norm_criterion(x_a) + norm_criterion(x_n)
        return loss
      
    loss = triplet_criterion(x_a, x_p, x_n)
    return loss
