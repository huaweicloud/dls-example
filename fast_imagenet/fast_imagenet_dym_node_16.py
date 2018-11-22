# coding:utf-8
#
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
# ==========================================================================

# python fast_imagenet_dym_node_16.py \
# --job_name=${JOB_NAME} \
# --data_url=/cache/ImageNet-shuffle \
# --data_url_160=/cache/ImageNet-160-shuffle \
# --bs_and_ims_strategy=20:128-224,38:224-224,40:288-128 \
# --fastai_initializer=False \
# --cooldown=0.08 \
# --local_parameter_device=gpu \
# --loss_scale=1024.0 \
# --max_lr=8.5 \
# --max_mom=0.98 \
# --min_lr=0.05 \
# --min_mom=0.9 \
# --num_gpus=8 \
# --num_inter_threads=14 \
# --num_intra_threads=14 \
# --num_readers=14 \
# --private_num_threads=24 \
# --num_prefetch_threads=24 \
# --split_dataset_like_mxnet=True \
# --running_log_level=2 \
# --save_model_secs=36000 \
# --save_summary_steps=30 \
# --server_protocol=grpc+verbs \
# --use_lars=False \
# --use_lr_schedule=lcd \
# --use_nesterov=True \
# --use_optimizer=dymomentum \
# --var_dtype=fp32 \
# --variable_update=distributed_replicated_mix \
# --min_size_in_kb=999999 \
# --var_space_starter=1024 \
# --var_mix_offset=10 \
# --log_every_n_steps=10 \
# --warmup_steps=0 \
# --warmup=0.1 \
# --weight_decay=0.0001 \
# --sync_before_server=True \
# --ps_hosts=${PS_HOSTS} \
# --worker_hosts=${WORKER_HOSTS} \
# --task_index=${TASK_INDEX}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
print('Training job start at: %s' % time.time())

import os
os.environ.pop('http_proxy', None)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import sys
import math

import tensorflow as tf
from tensorflow.contrib import slim

import moxing.tensorflow as mox
from moxing.tensorflow.datasets.tfrecord_image import ProgressiveImagenetDataset
from moxing.tensorflow.optimizer.dynamic_momentum import DyMomentumOptimizer
from moxing.tensorflow.optimizer.dynamic_momentumw import DyMomentumWOptimizer
from moxing.tensorflow.optimizer.learning_rate_scheduler import linear_cosine_decay, polynomial_decay


tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string('data_format',
                       default='NCHW', help='NHWC or NCHW')
tf.flags.DEFINE_string('data_url',
                       default=None, help='dataset dir')
tf.flags.DEFINE_string('data_url_160',
                       default='s3://wolfros-net/datasets/ImageNet-160', help='dataset dir')
tf.flags.DEFINE_integer('num_readers',
                        default=14, help='number of readers')
tf.flags.DEFINE_string('model_name',
                       default='resnet_v1_50_8k', help='model_name')
tf.flags.DEFINE_string('run_mode',
                       default='TRAIN', help='run_mode')
tf.flags.DEFINE_integer('batch_size',
                        default=256, help='batch size')
tf.flags.DEFINE_integer('log_every_n_steps',
                        default=10, help='log_every_n_steps')
tf.flags.DEFINE_float('weight_decay',
                      default=0.0001, help='Weight decay')
tf.flags.DEFINE_float('momentum', default=0.9,
                      help='Set 0 to use `SGD` opt, >0 to use momentum opt')
tf.flags.DEFINE_string('train_url',
                       default=None, help='Output url.')
tf.flags.DEFINE_string('checkpoint_url',
                       default=None, help='Output url.')
tf.flags.DEFINE_boolean('fp16',
                        default=True, help='Whether to use fp16 for the whole model')
tf.flags.DEFINE_string('local_cache',
                       default=None, help='In [hard, soft, None]')
tf.flags.DEFINE_integer('num_samples',
                        default=1281167, help='Number of samples per epoch.')
tf.flags.DEFINE_string('file_pattern',
                       default='train-*', help='File pattern when using tfrecord')
tf.flags.DEFINE_integer('save_summary_steps',
                        default=36000, help='')
tf.flags.DEFINE_integer('save_model_secs',
                        default=36000, help='')
tf.flags.DEFINE_string('learning_rate_strategy',
                       default=None, help='')
tf.flags.DEFINE_string('bs_and_ims_strategy',
                       default=None, help='39:224-256,40:288-128')
tf.flags.DEFINE_boolean('official_stride',
                        default=False, help='')
tf.flags.DEFINE_string('lr_strategy',
                       default='linear', help='')
tf.flags.DEFINE_boolean('fastai_initializer',
                        default=False, help='')
tf.flags.DEFINE_integer('private_num_threads',
                        default=24, help='')
tf.flags.DEFINE_boolean('split_dataset_like_mxnet', default=False, help='')
tf.flags.DEFINE_boolean('split_to_device', default=False, help='')
tf.flags.DEFINE_boolean('synthetic', default=False, help='')
tf.flags.DEFINE_boolean('gpu_synthetic',
                        default=False, help='')
tf.flags.DEFINE_string('cache_dir',
                       default=None, help='')
tf.flags.DEFINE_boolean('strict_sync_replicas', default=True, help='')
tf.flags.DEFINE_boolean('use_controller', default=False, help='')

tf.flags.DEFINE_string('use_optimizer', 'dymomentum', '')
tf.flags.DEFINE_string('use_lr_schedule', 'lcd', '')
tf.flags.DEFINE_boolean('use_nesterov', True, 'Whether to use nesterov accelerator.')
tf.flags.DEFINE_float('max_lr', 6.4, 'max learning rate.')
tf.flags.DEFINE_float('min_lr', 0.005, 'min learning rate.')
tf.flags.DEFINE_float('max_mom', 0.98, 'max momentum factor value in dynamic momentum optimizer.')
tf.flags.DEFINE_float('min_mom', 0.85, 'min momentum factor value in dynamic momentum optimizer.')
tf.flags.DEFINE_boolean('use_lars', False, 'Whether to use LARS.')
tf.flags.DEFINE_float('warmup', 0.1, 'The warmup ratio of steps.')
tf.flags.DEFINE_float('cooldown', 0.05, 'The cooldown ratio of steps.')

flags = tf.flags.FLAGS
flags(sys.argv, known_only=True)
# mox.set_flag('checkpoint_exclude_patterns', 'global_step')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def convert_ps_to_controller():
  # ps0 -> worker0
  # ps1 -> worker1
  # worker0 -> controller
  # worker1 -> sleep
  job_name = mox.get_flag('job_name')
  task_index = mox.get_flag('task_index')
  ps_hosts = mox.get_flag('ps_hosts')
  worker_hosts = mox.get_flag('worker_hosts')

  mox.set_flag('ps_hosts', '')
  mox.set_flag('worker_hosts', ps_hosts)
  mox.set_flag('controller_host', worker_hosts.split(',')[0])

  if job_name == 'ps':
    tf.logging.info('convert ps to worker')
    mox.set_flag('job_name', 'worker')
  elif job_name == 'worker' and task_index == 0:
    tf.logging.info('convert worker-0 to controller')
    mox.set_flag('job_name', 'controller')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  else:
    tf.logging.info('sleep unused server')
    time.sleep(9999999)


def download_dataset(data_url, data_url_160, skip_download=False):
  cache_url = '/cache/cache-imagenet'
  cache_url_160 = '/cache/cache-imagenet-160'
  if not skip_download:
    mox.file.copy_parallel(data_url, cache_url)
    mox.file.copy_parallel(data_url_160, cache_url_160)
  return cache_url, cache_url_160


def config_bs_ims(strategy):
  num_gpus = mox.get_flag('num_gpus')
  num_workers = len(mox.get_flag('worker_hosts').split(','))
  res = []
  if ":" not in strategy:
    image_size, batch_size = strategy.split('-')
    return [(float(image_size), float(batch_size))]
  else:
    stags = strategy.split(",")
    last_steps, last_epoch = 0, 0
    for i in range(len(stags)):
      cur_epoch, value = stags[i].strip().split(':')
      image_size, batch_size = value.strip().split('-')
      cur_epoch, image_size, batch_size = float(cur_epoch), float(image_size), float(batch_size)
      cur_batch_tot = batch_size * num_gpus * num_workers
      cur_steps = int(round(math.ceil(flags.num_samples / float(cur_batch_tot)))) * (
      cur_epoch - last_epoch) + last_steps
      res.append((int(cur_steps), int(image_size), int(batch_size)))
      last_steps, last_epoch = cur_steps, cur_epoch
  return res


def main(*args, **kwargs):
  if flags.use_controller:
    convert_ps_to_controller()

  job_name = mox.get_flag('job_name')
  task_index = mox.get_flag('task_index')

  if flags.local_cache == 'hard':
    if flags.use_controller:
      # In all-reduce mode, worker-0 does not download dataset (controller-0 will download).
      imagenet_data, imagenet_160_data = download_dataset(
        flags.data_url, flags.data_url_160,
        skip_download=(job_name == 'worker' and task_index == 0))
    else:
      # PS dose not download dataset.
      imagenet_data, imagenet_160_data = download_dataset(
        flags.data_url, flags.data_url_160,
        skip_download=(job_name == 'ps'))

    log_dir = '/cache/cache-outputs'
  else:
    imagenet_data = flags.data_url
    imagenet_160_data = flags.data_url_160
    log_dir = flags.train_url

  print('download dataset finish at %s' % time.time())

  if (not job_name or (job_name == 'worker' and task_index == 0)) and flags.train_url:
    if not mox.file.is_directory(log_dir):
      mox.file.make_dirs(log_dir)
  else:
    log_dir = None

  model_meta = mox.get_model_meta(flags.model_name)
  labels_offset = model_meta.default_labels_offset
  num_workers = len(mox.get_flag('worker_hosts').split(','))

  assert flags.bs_and_ims_strategy is not None
  schduler = config_bs_ims(flags.bs_and_ims_strategy)
  max_step = int(schduler[-1][0])

  def input_fn(mode, **kwargs):

    if not flags.synthetic:
      ds_strategy_spec = []
      ds_switch_steps = []

      if flags.split_dataset_like_mxnet and mox.get_flag('job_name'):
        if num_workers == 4:
          file_pattern = 'train-*-of-*-node-%d-*-*' % task_index
        elif num_workers == 8:
          file_pattern = 'train-*-of-*-node-*-%d-*' % task_index
        elif num_workers == 16:
          file_pattern = 'train-*-of-*-node-*-*-%d' % task_index
        else:
          raise ValueError('num_workers should be 4, 8, 16')

      else:
        file_pattern = flags.file_pattern

      for step, ims, bs in schduler:
        # switch to next dataset 2 steps earlier because there are 2 pipeline prefetch queue
        ds_switch_steps.append(step - 2)
        if ims == 128:
          ds_strategy_spec.append((os.path.join(imagenet_160_data, file_pattern), bs, ims, 0.08))
        elif ims == 224:
          ds_strategy_spec.append((os.path.join(imagenet_data, file_pattern), bs, ims, 0.087))
        elif ims == 288:
          ds_strategy_spec.append((os.path.join(imagenet_data, file_pattern), bs, ims, 0.5))
        else:
          raise ValueError('image is not in [128, 224, 288]')

      # The last stage of dataset does not need to be switched
      ds_switch_steps.pop(-1)
      tf.logging.info('Dataset will be switched at step: %s' % ds_switch_steps)

      dataset = ProgressiveImagenetDataset(
        num_samples=flags.num_samples,
        strategy_spec=ds_strategy_spec,
        ds_switch_steps=ds_switch_steps,
        shuffle=True, num_parallel=flags.num_readers,
        labels_offset=labels_offset,
        private_num_threads=flags.private_num_threads,
        shuffle_buffer_size=512 * 8 * 2)

      image, label = dataset.get(['image', 'label'])

      image_shape = tf.shape(image)[2]
      batch_size = tf.shape(label)[0]
      tf.summary.scalar(name='image_shape', tensor=image_shape)
      tf.summary.scalar(name='batch_size', tensor=batch_size)

    else:

      import numpy as np
      image = tf.constant(
        np.random.randint(low=0, high=255, size=[flags.batch_size, 128, 128, 3], dtype=np.uint8))
      label = tf.constant(
        np.random.randint(low=0, high=999, size=[flags.batch_size], dtype=np.int64))

    if flags.split_to_device:
      input_spec = mox.InputSpec(split_to_device=True)
      input_spec.new_input([image, label])
      return input_spec
    else:
      return image, label

  def model_fn(inputs, mode, **kwargs):
    if not flags.gpu_synthetic:
      if flags.split_to_device:
        images, labels = inputs.get_input(0)
      else:
        images, labels = inputs
    else:
      import numpy as np
      images = tf.constant(
        np.random.randint(low=0, high=255, size=[flags.batch_size, 128, 128, 3], dtype=np.uint8))
      labels = tf.constant(
        np.random.randint(low=0, high=999, size=[flags.batch_size], dtype=np.int64))

    if flags.fp16:
      images = tf.cast(images, tf.float16)

    def preprocess_fn(images, run_mode, *args):
      images = images / 255.0
      channels = tf.split(axis=3, num_or_size_splits=3, value=images)
      for i in range(3):
        channels[i] = (channels[i] - mean[i]) / std[i]
      images = tf.concat(axis=3, values=channels)
      if flags.data_format == 'NCHW':
        images = tf.transpose(images, perm=(0, 3, 1, 2))
      return images

    model_kwargs = {}
    if flags.model_name == 'resnet_v1_50_8k':
      if flags.official_stride:
        model_kwargs['official'] = True
      if flags.fastai_initializer:
        model_kwargs['weights_initializer_params'] = {'factor': 2.0 / 1.3, 'mode': 'FAN_OUT'}

    mox_model_fn = mox.get_model_fn(
      name=flags.model_name,
      run_mode=mode,
      num_classes=1000,
      preprocess_fn=preprocess_fn,
      weight_decay=flags.weight_decay,
      data_format=flags.data_format,
      batch_norm_fused=True,
      batch_renorm=False,
      **model_kwargs)

    logits, end_points = mox_model_fn(images)

    labels_one_hot = slim.one_hot_encoding(labels, 1000)
    loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels_one_hot,
      label_smoothing=0.0, weights=1.0)

    logits_fp32 = tf.cast(logits, tf.float32)
    accuracy_top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_fp32, labels, 1), tf.float32))
    accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_fp32, labels, 5), tf.float32))

    log_info = {'ent_loss': loss, 'top-1': accuracy_top_1, 'top-5': accuracy_top_5}

    regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(regularization_losses) > 0 and flags.use_optimizer != 'dymomentumw':
      regularization_loss = tf.add_n(regularization_losses)
      log_info['reg_loss'] = regularization_loss
      loss = loss + regularization_loss
      log_info['total_loss'] = loss

    return mox.ModelSpec(loss=loss, log_info=log_info)

  if flags.strict_sync_replicas:
    mox.set_flag('sync_replicas', False)
    mox.set_flag('chief_inc_global_step', True)

  def optimizer_fn():
    global_step = tf.train.get_or_create_global_step()
    decay_end = 1.0 - flags.cooldown

    if flags.use_lr_schedule == 'lcd':
      lr = linear_cosine_decay(flags.max_lr, flags.min_lr, global_step, max_step, flags.warmup,
                               decay_end)
      print("Using Linear Cosine Decay Schedule")
    elif flags.use_lr_schedule == 'poly':
      lr = polynomial_decay(flags.max_lr, flags.min_lr, global_step, max_step, flags.warmup,
                            decay_end)
      print("Using Polynomial Decay Schedule")
    else:
      raise ValueError("lr schedule not provided")

    if flags.use_optimizer == 'dymomentum':
      opt = DyMomentumOptimizer(lr, flags.max_lr, flags.min_lr, max_mom=flags.max_mom,
                                min_mom=flags.min_mom,
                                global_step=global_step, max_iteration=max_step,
                                use_nesterov=flags.use_nesterov,
                                cooldown=flags.cooldown, use_lars=flags.use_lars,
                                weight_decay=flags.weight_decay)
      print("Using Dynamic Momentum Optimizer")
    elif flags.use_optimizer == 'dymomentumw':
      opt = DyMomentumWOptimizer(lr, flags.max_lr, flags.min_lr, max_mom=flags.max_mom,
                                 min_mom=flags.min_mom,
                                 global_step=global_step, max_iteration=max_step,
                                 use_nesterov=flags.use_nesterov,
                                 cooldown=flags.cooldown, use_lars=flags.use_lars,
                                 weight_decay=flags.weight_decay)
      print("Using Dynamic MomentumW Optimizer")
    else:
      raise ValueError("Optimizer not provided")

    tf.summary.scalar(name='momentum', tensor=opt.get_momentum())

    if flags.strict_sync_replicas:
      from moxing.tensorflow.optimizer.simple_sync_optimizer import SimpleSyncOptimizer
      opt = SimpleSyncOptimizer(opt, num_workers=num_workers, task_index=task_index)

    return opt

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=optimizer_fn,
          run_mode=flags.run_mode,
          batch_size=flags.batch_size,
          max_number_of_steps=max_step,
          log_every_n_steps=flags.log_every_n_steps,
          log_dir=log_dir,
          auto_batch=False,
          save_summary_steps=flags.save_summary_steps,
          checkpoint_path=flags.checkpoint_url,
          save_model_secs=flags.save_model_secs)

  print('upload model finish at %s' % time.time())

  if flags.local_cache == 'hard' and log_dir:
    mox.file.copy_parallel(log_dir, flags.train_url)

  print('Training job finish at: %s' % time.time())

if __name__ == '__main__':
  tf.app.run(main=main)
