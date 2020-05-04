"""
  FileName     [ utils.py ]
  PackageName  [ final ]
  Synopsis     [ Utility functions for project. ]

  Usage:
  - To tide up the codes, add some frequency used function here.
    (by Edward Lee)
"""
import os
import torch 

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def load_network(network, file_path):
    """ 
      Load model parameters
    
      Params:
      - network: the instance of Nerual Network

      Return:
      - network: the instance of Nerual Network, with loaded parameter
    """
    network.load_state_dict(torch.load(file_path))

    return network

def fliplr(img):
    """
      Horizontal flip the image

      Params:
      - img: The image in torch.Tensor
    
      Returns:
      - img_filp: The image in torch.Tensor
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)

    return img_flip

def details(opt, fmt="{:16} {}", path=None):
    """
      Show and marked down the training settings

      Params:
      - opt: The namespace of the train setting (Usually argparser)
      - path: the path output textfile

      Return: None
    """
    for item, values in vars(opt).items():
        print(fmt.format(item, values))

    if isinstance(path, str):       
        makedirs = []
        folder = os.path.dirname(path)
        while not os.path.exists(folder):
            makedirs.append(folder)
            folder = os.path.dirname(folder)

        while len(makedirs) > 0:
            makedirs, folder = makedirs[:-1], makedirs[-1]
            os.makedirs(folder)

        with open(path, "w") as textfile:
            for item, values in vars(opt).items():
                msg = fmt.format(item, values)
                textfile.write(msg + '\n')
    
    return
