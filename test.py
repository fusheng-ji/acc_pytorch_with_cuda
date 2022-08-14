import torch
import cppcuda_tutorial
import time

def trilinear_interpolation_py(feats, points):
    feats = torch.ones(2)
    point = torch.zeros(2)

if __name__ == '__main__':
    feats = torch.ones(2, device='cuda')
    points = torch.zeros(2, device='cuda')

    out = cppcuda_tutorial.trilinear_interpolation(feats, points)

    print(out)

