from keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import threadsafe_generator
import os
import glob
from random import random
import tensorflow as tf
import cv2
import h5py
import numpy as np


class DataSet:
  # def __init__(self,batch_size):
  #   self.batch_size = self.batch_size


  @threadsafe_generator
  def generate(self, batch_size,flow_paths, flow_hdf5):
    flags = []
    flags = np.array(flags)
    # flow_paths = "F:/video_action_recognition/flow_output/test1"
    # frames = glob(os.path.join(flow_paths, '*.jpg'))
    frames = sorted(glob.glob(os.path.join(flow_paths, '*.jpg')), key=os.path.getmtime)
    for i, path in enumerate(frames):
      flag = path.split('.')
      flags = np.append(flags, int(flag[1]))
    x = DataSet.read(self,flow_hdf5)
    y = to_categorical(flags)
    x = tf.constant(x)
    y = tf.constant(y)
    # 批次给数据
    idx = 0
    x1 = []
    y1 = []
    while 1:
      idx += batch_size
      idy = idx + batch_size
      x1 = x[idx:idy]
      y1 = y[idx:idy]
      yield x1, y1

  def read(self,flow_hdf5):
    # with h5py.File('flow1.hdf5', 'r') as hf:
    with h5py.File(flow_hdf5, 'r') as hf:
      data = hf['mouse3_data'][:]
    data = np.array(data)
    data = np.delete(data, 0, axis=0)
    return data
