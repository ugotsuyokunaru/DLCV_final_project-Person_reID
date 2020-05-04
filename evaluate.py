"""
  FileName     [ evaluate.py ]
  PackageName  [ final ]
  Synopsis     [ Normalization function ]
"""

import torch
import numpy as np
from numpy import linalg as LA

def normalize_tensor(a: torch.Tensor, dim: int) -> torch.Tensor:
    """
      Normalize the tensor as unit vector
    """
    norm = torch.norm(a, dim=dim, keepdim=True)
    a = a / norm.expand_as(a)

    return a

def normalize_ndarray(a: np.ndarray, axis: int) -> np.ndarray:
    """
      Normalize the ndarray as unit vector
    """
    norm = LA.norm(a, axis=axis, keepdims=True)
    a = a / np.repeat(norm, a.shape[axis], axis=axis)
    
    return a

def cosine_similarity(cast_feature: torch.Tensor, cast_name: np.ndarray, candidate_feature: torch.Tensor, candidate_name: np.ndarray, mute=False) -> list:
    """
      Using cosine_similarity to sorting the query priorities.

      Return:
      - result: {'Id': 'Rank'} dicts in list
    """
    result = []

    cast_feature      = normalize_tensor(cast_feature, dim=1)
    candidate_feature = normalize_tensor(candidate_feature, dim=1)
    
    distance = torch.mm(cast_feature, candidate_feature.transpose(0, 1))
    index    = torch.argsort(distance, dim=1, descending=True).cpu().numpy()

    if not mute:
        print("Distance.shape: ", distance.shape)
        print("Min distance: ")
        print(list(map(lambda x: round(x, 4), distance.min(dim=1)[0].cpu().numpy().tolist())))     # Print values only, indices are deprecated
        print("Max distance: ")
        print(list(map(lambda x: round(x, 4), distance.max(dim=1)[0].cpu().numpy().tolist())))     # Print values only, indices are deprecated

    for i in range(index.shape[0]):
        result.append({
            'Id': cast_name[i],
            'Rank': ' '.join(candidate_name[index[i]])
        })

    return result
