"""
  FileName     [ utils.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Utility functions for project. ]

  Usage:
  - To tide up the codes, add some frequency used function here.
    (by Edward Lee)
"""
import torch 

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def details(opt, path=None):
    """
      Show and marked down the training settings

      Params:
      - opt: The namespace of the train setting (Usually argparser)
      - path: the path output textfile

      Return: None
    """
    for item, values in vars(opt).items():
        print("{:16} {}".format(item, values))

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
                msg = "{:16} {}".format(item, values)
                textfile.write(msg + '\n')
    
    return