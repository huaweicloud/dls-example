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
import moxing.pytorch as mox
import time
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.multiprocessing as mp

from torchvision import datasets, transforms
from moxing.pytorch.tools.AverageMeter import *
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None,
                    help='s3 path of dataset')
parser.add_argument('--train_url', type=str, default=None, help='s3 path of outputs')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=10, type=int,
                    help='The epochs for training')
parser.add_argument('--num_workers', type=int, default=2,
                    help='the number of workers of dataloader')
parser.add_argument('--print_freq', type=int, default=1, help='')
parser.add_argument('--mp', type=bool, default=False, help='Multiprocess distributed mode')


def main_worker(gpu, args):

  mox.dist.init_process_group(mp=args.mp, gpu=gpu)

  # Auto Model
  model = torchvision.models.resnet50()
  model = mox.dist.AutoModule(model)

  # Auto Dataloader
  data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  image_datasets = datasets.ImageFolder(args.data_url, data_transforms)
  train_set = mox.dist.AutoDataLoader(image_datasets, pin_memory=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)

  criterion = nn.CrossEntropyLoss().cuda()

  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  for epoch in range(args.epochs):
    train_set.set_epoch(epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch train mode
    scheduler.step()
    model.train()

    epoch_begin = time.time()
    end = time.time()
    for i, (data, target) in enumerate(train_set):
      data_time.update(time.time() - end)

      # # forward
      if not args.mp and torch.cuda.is_available():
        target = target.cuda(non_blocking=True)
      outputs = model(data)
      loss = criterion(outputs, target)
      losses.update(loss.item(), data.size(0))

      # backward + optimizer
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        if gpu is None or gpu == 0:
          print('Epoch: [{0}]\t'
                'steps: {1}\t'
                'Sample/secs: {2:.2f}\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            epoch, i, args.batch_size / batch_time.val, batch_time=batch_time,
            data_time=data_time, loss=losses))

    print('The {} epoch use {} seconds for training'.format(epoch, time.time() - epoch_begin))
    # empty temporary variable
    torch.cuda.empty_cache()


if __name__ == "__main__":
  # Protect the arguments which are not parsed.
  args, unparsed = parser.parse_known_args()

  if args.mp:
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(),
             args=(args,))
  else:
    main_worker(None, args)

