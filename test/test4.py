import os

# import h5py
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from numpy import set_printoptions

_IMAGE_SIZE = 256


# def cal_for_frames(video_path):
def cal_for_frames(video_path,flow_path):
  # frames = glob(os.path.join(video_path, '*.jpg'))
  frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=os.path.getmtime)
  # print(frames)
  print(np.array(frames).shape)
  flow = []
  sum = 0
  prev = cv2.imread(frames[0])
  # cv2.imshow("frames",frames[28])
  prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
  for i, frame_curr in enumerate(frames):
    curr = cv2.imread(frame_curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    tmp_flow = compute_TVL1(prev, curr)
    # cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
    #             flow[:, :, 0])
    # cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
    #             flow[:, :, 1])
    #计算各帧间光流矢量之和并归一化及可视化
    for cow,flow2 in enumerate(tmp_flow):
      for column,flow1 in enumerate(flow2):
        a=(((tmp_flow[cow,column,0]**2+tmp_flow[cow,column,1]**2)**0.5))

        sum+=a
    # print((sum/16000000-1)*100,i)
    flow.append(sum)
    sum = 0
    prev = curr
    print(i)
  flow_1 = np.array(flow).astype(float)
  flow_std=(flow_1-flow_1.min(axis=0))/(flow_1.max(axis=0)-flow_1.min(axis=0))
  np.set_printoptions(threshold=1000)
  print(flow_std)
  X = np.linspace(0, flow_std.shape[0]-1, flow_std.shape[0])
  Y=flow_std
  T = np.arctan2(Y, X)
  plt.scatter(X, Y, s=75, c=T, alpha=.5)

  # plt.xlim(-1.5, 1.5)
  # plt.xticks(())  # ignore xticks
  # plt.ylim(-1.5, 1.5)
  # plt.yticks(())  # ignore yticks

  plt.show()
  # func = np.poly1d(flow_std)
  # func1 = func.deriv(m=1)
  # x = np.linspace(0,flow_std.shape[0],flow_std.shape[0])
  # y=func(x)
  # y1 = func1(x)
  # plt.plot(x, y, 'ro', x, y1, 'g--')
  # plt.xlabel('x')
  # plt.ylabel('y')
  # plt.show()
  # print(flow_std.shape[0])
  # print(flow_std[9])


  # with h5py.File("flow.hdf5", 'w') as f:
  #   f.create_dataset("a_data", data=flow, compression="gzip", compression_opts=5)
  # return flow



def compute_TVL1(prev, curr, bound=15):
  """Compute the TV-L1 optical flow."""
  # TVL1 = cv2.flowDualTVL1OpticalFlow_create()
  TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
  # TVL1 = cv2.DualTVL1OpticalFlow_create()
  # TVL1=cv2.createOptFlow_DualTVL1()
  flow = TVL1.calc(prev, curr, None)
  assert flow.dtype == np.float32
  flow = (flow + bound) * (255.0 / (2 * bound))
  flow[flow >= 255] = 255
  flow[flow <= 0] = 0
  return flow


# def save_flow(video_flows, flow_path):
#   for i, flow in enumerate(video_flows):
#     cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
#                 flow[:, :, 0])
#     cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
#                 flow[:, :, 1])
#
#
# def extract_flow(video_path, flow_path):
#   flow = cal_for_frames(video_path)
#   save_flow(flow, flow_path)
#   print('complete:' + flow_path)
#   return

  # def calu(flow):
  #   for vector in enumerate(flow):
  #     a=vector[:,:,0]
  #     b=vector[:,:,1]



if __name__ == '__main__':
  # video_paths = "/home/xueqian/bishe/extrat_feature/output"
  # flow_paths = "/home/xueqian/bishe/extrat_feature/flow"
  video_paths = "F:/video_action_recognition/frame_output/mouse1"
  flow_paths = "F:/video_action_recognition/flow_output/test1"
  video_lengths = 109

  cal_for_frames(video_paths, flow_paths)
