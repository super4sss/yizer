import csv
import os
from glob import glob
from random import random

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
# def save():
  # video_path = "F:/video_action_recognition/frame_output/mouse1"
  # frames = glob(os.path.join(video_path, '*.jpg'))
  # frames = np.array(frames)
  # data=cv2.imread(frames[0])
  # print(data.shape)
  # # with open('flow.csv', 'w', newline='') as f:
  # #   writer = csv.writer(f)
  # #   writer.writerows(data)

def read():
  with h5py.File('flow1.hdf5', 'r') as hf:
    data = hf['a_data'][:]
  data=np.array(data)
  # data = tf.expand_dims(data,4)
  data=np.delete(data, 0, axis=0)
  print(data.shape)

def read1():
  j=1
  flags=[]
  flags=np.array(flags)
  flow_paths = "F:/video_action_recognition/flow_output/test1"
  frames = glob(os.path.join(flow_paths, '*.jpg'))
  for i, path in enumerate(frames):
    flag=path.split('.')
    flags=np.append(flags,int(flag[1]))
  print(flags)


def main():
  read1()

if __name__ == '__main__':
  main()
