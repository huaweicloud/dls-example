

# coding:utf-8

# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import moxing.pytorch as mox
from moxing.pytorch.executor.enumerate import *


# python imagenet_mox.py \
# --data_url=/cache/ilsvrc \
# --num_workers=14 \
# --batch_size=1024
# --world_size=2
# --rank=0
# --init_method='tcp://169.254.128.124:40000'

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default='/cache/tiny-imagenet-200/train',
                    help='s3 path of dataset')
parser.add_argument('--train_url', type=str, default=None, help='s3 path of outputs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--num_workers', type=int, default=14,
                    help='the number of workers of dataloader')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--num_classes', type=int, default=1000, help='thr number of classes')
parser.add_argument('--print_freq', type=int, default=10, help='')

parser.add_argument('--init_method', type=str, default=None, help='master address')
parser.add_argument('--rank', type=int, default=0, help='Index of current task')
parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()


def loss_fn(model, inputs):
  x, y = inputs
  y = y.cuda()
  outputs = model(x)
  criterion = nn.CrossEntropyLoss()
  loss = criterion(outputs, y)
  return loss


def main():

  mox.dist.init_process_group()

  data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  image_datasets = datasets.ImageFolder(args.data_url, data_transforms)
  train_set = torch.utils.data.DataLoader(image_datasets, shuffle=False, pin_memory=True,
                                       num_workers=args.num_workers, batch_size=args.batch_size)

  model = torchvision.models.resnet50(pretrained=False)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)

  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  mox.run(input_data=train_set,
          model=model,
          run_mode=ModeKeys.TRAIN,
          log_dir='/home/lee/var',
          optimizer=optimizer,
          scheduler=scheduler,
          loss_fn=loss_fn,
          epoches=1,
          num_gpus=4,
          print_freq=128)

if __name__ == "__main__":
  #if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 4):
   # raise ValueError('PyTorch distributed running with `DistributedDataParallel` '
    #                 'only support python >= 3.4')
  #mp.set_start_method('spawn')
  # os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
  # mp.set_start_method('spawn')
  # mox.set_flag('rank', 0)
  main()

