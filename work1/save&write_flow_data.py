import csv
import os

import h5py
import numpy as np
import cv2
import glob

_IMAGE_SIZE = 256



# 将光流图像保存到hdf5文件中

def cal_for_frames(video_path,flow_paths):
  # frames = glob(os.path.join(video_path, '*.jpg'))
  frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=os.path.getmtime)
  print(np.array(frames).shape)
  flow = []
  prev = cv2.imread(frames[0])
  prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
  for i, frame_curr in enumerate(frames):
    curr = cv2.imread(frame_curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    tmp_flow = compute_TVL1(prev, curr)
    cv2.imwrite(os.path.join(flow_paths.format('u'), "{:06d}.jpg".format(i)),
                tmp_flow[:, :, 0])
    cv2.imwrite(os.path.join(flow_paths.format('v'), "{:06d}.jpg".format(i)),
                tmp_flow[:, :, 1])
    flow.append(tmp_flow)
    prev = curr
    print(i)
  print(np.array(flow).shape)
  flow =np.array(flow)
  return flow

def compute_TVL1(prev, curr, bound=15):
  """Compute the TV-L1 optical flow."""
  # TVL1 = cv2.flowDualTVL1OpticalFlow_create()
  TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
  # TVL1 = cv2.DualTVL1OpticalFlow_create()
  # TVL1=cv2.createOptFlow_DualTVL1()
  flow = TVL1.calc(prev, curr, None)
  assert flow.dtype == np.float32
  flow = (flow + bound) * (255.0 / (2 * bound))
  flow = np.round(flow).astype(int)
  flow[flow >= 255] = 255
  flow[flow <= 0] = 0

  return flow



# def save(video_path):
#   flow =cal_for_frames(video_path)
#   np.save("flow.csv",flow,delimiter=",",fmt="%s")

# import h5py
def save(video_path,flow_paths):  # 写入的时候是‘w’
  flow = cal_for_frames(video_path,flow_paths)
  with h5py.File("mouse3.hdf5", 'w') as f:
    f.create_dataset("mouse3_data", data=flow, compression="gzip", compression_opts=5)
  # f.create_dataset("b_data", data=b, compression="gzip", compression_opts=5)
  # f.create_dataset("c_data", data=c, compression="gzip", compression_opts=5)
  # with open('flow.csv', 'w', newline='') as f:  # 这里的csv则是最后输出得到的新表
  #   data = cal_for_frames(video_path)
  #   writer = csv.writer(f)
  #   # a = [[row] for row in data]
  #   # writer.writerows(a)
  #   writer.writerows(data)
  # data = cal_for_frames(video_path)
  # np.savetxt('flow.csv', data, fmt='%s%s%s%s', delimiter=',')



if __name__ == '__main__':
  # video_paths = "/home/xueqian/bishe/extrat_feature/output"
  # flow_paths = "/home/xueqian/bishe/extrat_feature/flow"
  video_paths = "F:/video_action_recognition/frame_output/v4mouse3"
  flow_paths = "F:/video_action_recognition/flow_output/v4mouse3"
  # video_lengths = 109

  save(video_paths,flow_paths)
