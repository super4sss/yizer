import os
import glob
import random
from random import Random

import h5py
from keras.optimizers import SGD
from keras.utils import to_categorical

from work1 import cnn_model_v4
from work1.cnn_model import ResearchModels
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = ""
def main():
  num_of_snip = 1
  opt_flow_len = 2
  saved_model = None
  class_limit = None
  image_shape = (280, 320)
  load_to_memory = False
  batch_size = 1
  nb_epoch = 6
  name_str = None

  # Get the model.
  # temporal_cnn = ResearchModels(nb_classes=2, num_of_snip=num_of_snip, opt_flow_len=opt_flow_len,
  #                               image_shape=image_shape, saved_model=saved_model)
  temporal_cnn=cnn_model_v4.inception_v4_backbone()
  # y = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
  # y=[]
  def generate():
    flags = []
    flags = np.array(flags)
    flow_paths1 = "F:/video_action_recognition/flow_output/v4mouse1"
    flow_paths2 = "F:/video_action_recognition/flow_output/v4mouse1"
    # frames = glob(os.path.join(flow_paths, '*.jpg'))
    frames1 = sorted(glob.glob(os.path.join(flow_paths1, '*.jpg')), key=os.path.getmtime)
    frames2 = sorted(glob.glob(os.path.join(flow_paths2, '*.jpg')), key=os.path.getmtime)
    frames1.extend(frames2)
    for i, path in enumerate(frames1):
      flag = path.split('.')
      flags = np.append(flags, int(flag[1]))
    x=read()
    y = to_categorical(flags)
    x=tf.constant(x)
    y=tf.constant(y)
    #批次给数据
    idx = 0
    x1 = []
    y1 = []
    while 1:
      # if idx==330:
      #   idx=0
      idx += batch_size
      idy=idx+batch_size
      x1=x[idx:idy]
      y1=y[idx:idy]

      # r=random.randint(0, 360)
      # s=random.randint(0, 360)
      # x1.append(x[r])
      # x1.append(x[s])
      # y1.append(y[r])
      # y1.append(y[s])

      yield x1, y1


  # checkpoint
  filepath = "model/weights.{epoch:02d}-{loss:.4f}-{accuracy:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False,
                               mode='auto', period=1)
  callbacks_list = [checkpoint]

  #complile
  # Set the metrics. Only use top k if there's a need.
  metrics = ['accuracy']


  # optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)
  optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

  temporal_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
  # self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

  print(temporal_cnn.summary())





  #---------------------------


  # temporal_cnn.model.fit_generator(
  temporal_cnn.fit_generator(

    # X=read().tolist(),
    generator=generate(),
    # validation_split=0.33,
    # batch_size=batch_size,
    # verbose=1,
    # verbose=0,
    # batch_size=1,
    epochs=10,
    steps_per_epoch=10,
    callbacks = callbacks_list,
  )



def read():
  # with h5py.File('v4flow.hdf5', 'r') as hf:
  #   data = hf['mouse1_data'][:]
  # data=np.array(data)
  # data=np.delete(data,0,axis=0)
  # # data = data[1:361]
  # return data
  with h5py.File('mouse1.hdf5', 'r') as hf:
    data1 = hf['mouse1_data'][:]
  with h5py.File('mouse2.hdf5', 'r') as hf:
    data2 = hf['mouse2_data'][:]
  data1 = np.delete(data1, 0, axis=0).tolist()
  data2 = np.delete(data2, 0, axis=0).tolist()
  data1.extend(data2)
  return data1

if __name__ == '__main__':
  main()
