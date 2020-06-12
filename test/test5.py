import h5py
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob

with h5py.File("flow.hdf5", 'r') as f:  # 读取的时候是‘r’
  print(f.keys())
  flow = f.get("a_data")[:]
  print(flow.shape)
  flow = flow[52:86:1]
  sumArr = []
  sum=0
  print(flow)
  for i, flow1 in enumerate(flow):
    for cow, flow2 in enumerate(flow1):
      for column, flow3 in enumerate(flow2):
        a = (((flow1[cow, column, 0] ** 2 + flow1[cow, column, 1] ** 2) ** 0.5))
        sum += a
    sumArr.append(sum)
    sum=0
  flow_1 = np.array(sumArr).astype(float)
  flow_std = (flow_1 - flow_1.min(axis=0)) / (flow_1.max(axis=0) - flow_1.min(axis=0))
  np.set_printoptions(threshold=1000)
  print(flow_std)
  X = np.linspace(0, flow_std.shape[0] - 1, flow_std.shape[0])
  Y = flow_std
  T = np.arctan2(Y, X)
  plt.scatter(X, Y, s=75, c=T, alpha=.5)
  plt.show()
