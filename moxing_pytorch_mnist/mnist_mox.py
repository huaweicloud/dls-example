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

from moxing.pytorch.executor.enumerate import *
from torch.optim import lr_scheduler
from torch.utils.data import distributed

import torch
import moxing.pytorch as mox
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import multiprocessing as mp
import torch.nn.functional as F

def load_data():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
  trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True,
    num_workers=10, pin_memory=True, sampler=None)
  return train_loader

def loss_fn(model, inputs):
  rank = mox.get_flag('rank')
  x, y = inputs
  x = x.cuda()
  y = y.cuda()
  outputs = model(x)
  loss = F.nll_loss(outputs, y)
  return loss

class Net(nn.Module):
  """ Network architecture. """

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=3)
    self.conv2 = nn.Conv2d(1, 1, kernel_size=3)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(25, 5)
    self.fc2 = nn.Linear(5, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 25)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

if __name__ == '__main__':
  mox.dist.init_process_group()
  dataset = load_data()
  model = Net()
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  mox.run(input_data=dataset,
          model=model,
          run_mode=ModeKeys.TRAIN,
          log_dir='/var/log/tmp',
          optimizer=optimizer,
          scheduler=scheduler,
          loss_fn=loss_fn,
          epoches=2,
          print_freq=128)

