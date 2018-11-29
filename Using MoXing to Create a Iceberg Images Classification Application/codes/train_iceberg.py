from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ.pop('http_proxy', None)
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import moxing.tensorflow as mox
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.python.keras.layers import Dropout, Flatten, Activation, Concatenate
slim = tf.contrib.slim
NUM_SAMPLES_TRAIN = 1176
NUM_SAMPLES_EVAL = 295
NUM_SAMPLES_TEST = 8424
tf.flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Dir of log')
tf.flags.DEFINE_boolean('is_training', True, 'True for train. False for eval and predict.')
flags = tf.flags.FLAGS

import atexit
import logging
_data_url = flags.data_url
_train_url = flags.train_url
if not mox.file.is_directory(_train_url):
  mox.file.make_dirs(_train_url)
mox.file.make_dirs('/cache/data_url')
mox.file.make_dirs('/cache/train_url')
mox.file.copy_parallel(_data_url, '/cache/data_url')
mox.file.copy_parallel(_train_url, '/cache/train_url')
flags.data_url = '/cache/data_url'
flags.train_url = '/cache/train_url'
atexit.register(lambda: mox.file.copy_parallel('/cache/train_url', _train_url))
logger = logging.getLogger()
while logger.handlers:
  logger.handlers.pop()

num_gpus = mox.get_flag('num_gpus')
num_workers = len(mox.get_flag('worker_hosts').split(','))
steps_per_epoch = int(math.ceil(float(NUM_SAMPLES_TRAIN) / (flags.batch_size * num_gpus * num_workers)))
submission = pd.DataFrame(columns=['id', 'is_iceberg'])
def input_fn(run_mode, **kwargs):
  if run_mode == mox.ModeKeys.TRAIN:
    num_samples = NUM_SAMPLES_TRAIN
    num_epochs = None
    shuffle = True
    file_pattern = 'iceberg-train-*.tfrecord'
  else:
    num_epochs = 1
    shuffle = False
    if run_mode == mox.ModeKeys.EVAL:
      num_samples = NUM_SAMPLES_EVAL
      file_pattern = 'iceberg-eval-*.tfrecord'
    else:
      num_samples = NUM_SAMPLES_TEST
      file_pattern = 'iceberg-test-*.tfrecord'
  keys_to_features = {
    'band_1': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'band_2': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'angle': tf.FixedLenFeature([1], tf.float32, default_value=None),
  }
  items_to_handlers = {
    'band_1': slim.tfexample_decoder.Tensor('band_1', shape=[75, 75]),
    'band_2': slim.tfexample_decoder.Tensor('band_2', shape=[75, 75]),
    'angle': slim.tfexample_decoder.Tensor('angle', shape=[])
  }
  if run_mode == mox.ModeKeys.PREDICT:
    keys_to_features['id'] = tf.FixedLenFeature([1], tf.string, default_value=None)
    items_to_handlers['id'] = slim.tfexample_decoder.Tensor('id', shape=[])
  else:
    keys_to_features['label'] = tf.FixedLenFeature([1], tf.int64, default_value=None)
    items_to_handlers['label'] = slim.tfexample_decoder.Tensor('label', shape=[])
  dataset = mox.get_tfrecord(dataset_dir=flags.data_url,
                             file_pattern=file_pattern,
                             num_samples=num_samples,
                             keys_to_features=keys_to_features,
                             items_to_handlers=items_to_handlers,
                             num_epochs=num_epochs,
                             shuffle=shuffle)
  if run_mode == mox.ModeKeys.PREDICT:
    band_1, band_2, id_or_label, angle = dataset.get(['band_1', 'band_2', 'id', 'angle'])
    # Non-DMA safe string cannot tensor may not be copied to a GPU.
    # So we encode string to a list of integer.
    id_or_label = tf.py_func(lambda str: np.array([ord(ch) for ch in str]), [id_or_label], tf.int64)
    # We know `id` is a string of 8 alphabets.
    id_or_label = tf.reshape(id_or_label, shape=(8,))
  else:
    band_1, band_2, id_or_label, angle = dataset.get(['band_1', 'band_2', 'label', 'angle'])
  band_3 = band_1 + band_2
  # Rescale the input image to [0, 1]
  def rescale(*args):
    ret_images = []
    for image in args:
      image = tf.cast(image, tf.float32)
      image_min = tf.reduce_min(image)
      image_max = tf.reduce_max(image)
      image = (image - image_min) / (image_max - image_min)
      ret_images.append(image)
    return ret_images
  band_1, band_2, band_3 = rescale(band_1, band_2, band_3)
  image = tf.stack([band_1, band_2, band_3], axis=2)
  # Data augementation
  if run_mode == mox.ModeKeys.TRAIN:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random_uniform(shape=(), maxval=3, minval=0, dtype=tf.int32))
  return image, id_or_label, angle
def model_v1(images, angles, run_mode):
  is_training = (run_mode == mox.ModeKeys.TRAIN)
  # Conv Layer 1
  x = Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3))(images)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)
  # Conv Layer 2
  x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)
  # Conv Layer 3
  x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)
  # Conv Layer 4
  x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)
  # Flatten the data for upcoming dense layers
  x = Flatten()(x)
  x = Concatenate()([x, angles])
  # Dense Layers
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x, training=is_training)
  # Dense Layer 2
  x = Dense(256)(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x, training=is_training)
  # Sigmoid Layer
  logits = Dense(2)(x)
  return logits
def model_fn(inputs, run_mode, **kwargs):
  # In train or eval, id_or_labels represents labels. In predict, id_or_labels represents id.
  images, id_or_labels, angles = inputs
  # Reshape angles from [batch_size] to [batch_size, 1]
  angles = tf.expand_dims(angles, 1)
  # Apply your version of model
  logits = model_v1(images, angles, run_mode)
  if run_mode == mox.ModeKeys.PREDICT:
    logits = tf.nn.softmax(logits)
    # clip logits to get lower loss value.
    logits = tf.clip_by_value(logits, clip_value_min=0.05, clip_value_max=0.95)
    model_spec = mox.ModelSpec(output_info={'id': id_or_labels, 'logits': logits})
  else:
    labels_one_hot = slim.one_hot_encoding(id_or_labels, 2)
    loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels_one_hot,
      label_smoothing=0.0, weights=1.0)
    model_spec = mox.ModelSpec(loss=loss, log_info={'loss': loss})
  return model_spec
def output_fn(outputs):
  global submission
  for output in outputs:
    for id, logits in zip(output['id'], output['logits']):
      # Decode id from integer list to string.
      id = ''.join([chr(ch) for ch in id])
      # Get the probability of label==1
      is_iceberg = logits[1]
      df = pd.DataFrame([[id, is_iceberg]], columns=['id', 'is_iceberg'])
      submission = submission.append(df)
if __name__ == '__main__':
  if flags.is_training:
    mox.run(input_fn=input_fn,
            model_fn=model_fn,
            optimizer_fn=mox.get_optimizer_fn(name='adam', learning_rate=0.001),
            run_mode=mox.ModeKeys.TRAIN,
            batch_size=flags.batch_size,
            log_dir=flags.train_url,
            max_number_of_steps=steps_per_epoch * 150,
            log_every_n_steps=20,
            save_summary_steps=50,
            save_model_secs=120)
  else:
    mox.run(input_fn=input_fn,
            model_fn=model_fn,
            run_mode=mox.ModeKeys.EVAL,
            batch_size=5,
            log_every_n_steps=1,
            max_number_of_steps=int(NUM_SAMPLES_EVAL / 5),
            checkpoint_path=flags.train_url)
    mox.run(input_fn=input_fn,
            output_fn=output_fn,
            model_fn=model_fn,
            run_mode=mox.ModeKeys.PREDICT,
            batch_size=24,
            max_number_of_steps=int(NUM_SAMPLES_TEST / 24),
            log_every_n_steps=50,
            output_every_n_steps=1,
            checkpoint_path=flags.train_url)
    # Write results to file. tf.gfile allow writing file to EBS/s3
    submission_file = os.path.join(flags.train_url, 'submission.csv')
    result = submission.to_csv(path_or_buf=None, index=False)
    with tf.gfile.Open(submission_file, 'w') as f:
      f.write(result)
