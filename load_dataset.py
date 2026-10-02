# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                     mean_subtraction=True)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import cifar10

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/tmp/nngp/data/',
                    'Directory for data.')

def load_mnist(num_train=50000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  """Loads MNIST as numpy array."""

  data_dir = FLAGS.data_dir
  print('****************data_dir***************:',data_dir)
  datasets = input_data.read_data_sets(
      data_dir, False, validation_size=10000, one_hot=True)
  mnist_data = _select_mnist_subset(
      datasets,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return mnist_data


def _select_mnist_subset(datasets,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  subset = copy.deepcopy(datasets)

  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(subset.train.labels, axis=1)  # undo one-hot

  for digit in digits:
    if datasets.train.num_examples == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = subset.train.images[idx_list][:num_train].astype(data_precision)
  train_label = subset.train.labels[idx_list][:num_train].astype(data_precision)
  valid_image = subset.validation.images.astype(data_precision)
  valid_label = subset.validation.labels.astype(data_precision)
  test_image = subset.test.images.astype(data_precision)
  test_label = subset.test.labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)

""" *********** Beginning of modification *************"""
def load_cifar10(num_train, mean_subtraction=True, num_valid=5000):
  """Loads CIFAR-10 as flattened, one-hot numpy arrays.

  Not part of the original repo's load_dataset.py (which only implements
  MNIST) -- added here so this file is a self-contained drop-in and doesn't
  require editing load_dataset.py. Mirrors load_dataset.load_mnist's
  signature and output shapes: [N, 3072] float inputs, [N, 10] one-hot
  labels, with train/valid/test splits.
  """
  # Bundled with TF 1.15's Keras; downloads to ~/.keras/datasets on first
  # use, same as load_mnist downloading via input_data.read_data_sets.
  # from tensorflow.keras.datasets import cifar10  # pylint: disable=g-import-not-at-top

  (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

  def _flatten(x):
    return x.reshape(x.shape[0], -1).astype(np.float64) / 255.0

  def _one_hot(y, num_classes=10):
    y = y.reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

  x_train_full = _flatten(x_train_full)
  x_test = _flatten(x_test)
  y_train_full = _one_hot(y_train_full)
  y_test = _one_hot(y_test)

  if num_train + num_valid > x_train_full.shape[0]:
    raise ValueError(
        'num_train (%d) + validation holdout (%d) exceeds the CIFAR-10 '
        'training set size (%d).' % (
            num_train, num_valid, x_train_full.shape[0]))

  train_image = x_train_full[:num_train]
  train_label = y_train_full[:num_train]
  valid_image = x_train_full[-num_valid:]
  valid_label = y_train_full[-num_valid:]

  if mean_subtraction:
    mean = train_image.mean(axis=0)
    train_image = train_image - mean
    valid_image = valid_image - mean
    x_test = x_test - mean

  return train_image, train_label, valid_image, valid_label, x_test, y_test
""" ********* End of modification ************* """

