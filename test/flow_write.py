import os
import numpy as np
import cv2
import glob

_IMAGE_SIZE = 256


# def cal_for_frames(video_path):
def cal_for_frames(video_path):
  frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=os.path.getmtime)
  # frames.sort()
  # print(np.array(frames).shape)
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



def calc(video_path):
  flow =cal_for_frames(video_path)
  for i,flow3 in enumerate(flow) :
    sum=0
    for cow,flow2 in enumerate(flow3):
      for column,flow1 in enumerate(flow2):
        sum+=(((flow[i,cow,column,0]**2+flow[i,cow,column,1]**2)**0.5)/16000)
    print(sum)





if __name__ == '__main__':
  # video_paths = "/home/xueqian/bishe/extrat_feature/output"
  # flow_paths = "/home/xueqian/bishe/extrat_feature/flow"
  video_paths = "F:/video_action_recognition/frame_output/test2"
  # flow_paths = "F:/video_action_recognition/flow_output/test1"
  video_lengths = 109

  calc(video_paths)
