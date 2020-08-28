import flop_counter
import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str, timing
# from models import print_log
import models
import random
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    model = flop_counter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    inp = torch.randn(*input_size)
    out = model(inp)
    flops = model.compute_average_flops_cost()
    return flops

model = models.__dict__['resnet50'](pretrained='use_pretrain')
model = torch.nn.DataParallel(model).cuda()
total = sum([param.nelement() for param in model.parameters()])
print('Number of full params: %.2fM' % (total / 1e6))     #每一百万为一个单位
checkpoint = torch.load(
    '/home/liuzili/FPKM/snapshots_imagenet/resnet50_d4_03/best.resnet50.2020-08-22-7348.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

full_flops = calculate_flops(model)
print('Full model FLOPS = %.4f (M)' % (full_flops / 1e6))


total = 0
for param in model.parameters():
    # if len(param.size()) == 4:
    total += int(torch.sum(param!=0))
print('Number of pruned params: %.2fM' % (total / 1e6))
# lr = []
# for epoch in range(100):
#     if epoch <= 100-10 and epoch > 30:
#         min_lr = 0.001
#         tmp_lr = min_lr + 0.5*(0.1-min_lr)*(1+math.cos(math.pi*(epoch-30)*1./\
#                         (100-10-30)))
#         lr.append(tmp_lr)
#     elif epoch > 100-10:
#         tmp_lr = 0.001
#         lr.append(tmp_lr)
#     else:
#         tmp_lr = 0.1
#         lr.append(tmp_lr)
# plt.plot(lr)
# plt.show()