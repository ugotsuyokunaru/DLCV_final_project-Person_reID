"""
  FileName     [ norm_loss.py ]
  PackageName  [ final ]
  Synopsis     [ Loss function of norm vector penalty ]

  - torch.nn.functional.normalize(input, dim=1)
  - torch.nn.functional.mse_loss(input, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormLoss(nn.Module):
    def __init__(self, lambda_value=5e-5):
        super(NormLoss, self).__init__()
        self.lambda_value = lambda_value

    def forward(self, x):
        unit_vector = F.normalize(x, dim=1)
        loss = self.lambda_value * F.mse_loss(x, unit_vector)

        return loss

class TestNet(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def unittest():
    net = TestNet()
    x = torch.randn(8, 2048)

    criterion = NormLoss(1e-5)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    
    for i in range(0, 100):
        optimizer.zero_grad()
        y = net(x)

        loss = criterion(y)
        loss.backward()
        optimizer.step()

        # The norm of y is decreasing.
        print(torch.norm(y, dim=1))

    return

def main():
    unittest()

if __name__ == '__main__':
    main()