import os
import numpy as np
import cv2
from glob import glob

_IMAGE_SIZE = 256


# def cal_for_frames(video_path):
def cal_for_frames(video_path):
  frames = glob(os.path.join(video_path, '*.jpg'))
  frames.sort()
  print(np.array(frames).shape)
  flow = []
  prev = cv2.imread(frames[0])
  prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
  for i, frame_curr in enumerate(frames):
    curr = cv2.imread(frame_curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    tmp_flow = compute_TVL1(prev, curr)
    flow.append(tmp_flow)

    prev = curr
    print(i)
  print(np.array(flow).shape)
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


def save_flow(video_flows, flow_path):
  for i, flow in enumerate(video_flows):
    cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
                flow[:, :, 0])
    cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
                flow[:, :, 1])


def extract_flow(video_path, flow_path):
  flow = cal_for_frames(video_path)
  save_flow(flow, flow_path)
  print('complete:' + flow_path)
  return

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

  extract_flow(video_paths, flow_paths)
