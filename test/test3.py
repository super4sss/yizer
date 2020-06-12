import os
import glob

# import h5py
# import tensorflow as tf
import numpy as np
import imageio
import cv2

from PIL import Image
# b=tf.random.uniform([2,2], minval=0, maxval=10, dtype=tf.int32)
# print(b)
# c=tf.reshape(b,[-1,4])
# print(c)

# y = tf.constant([1,2]) # 数字编码
# s = tf.one_hot(y, depth=4) # one-hot 编码
# # s = tf.reshape(y,[1,20])
# print(s)
# i=1
# print('_%d.jpg' %i)
# for i in range(1,10,1):
#   while(i%2==0):
#     print(i)
#     break
video_path = "F:/video_action_recognition/flow_output/mouse1"
# frames = glob(os.path.join(video_path, '*.jpg'))
# frames.sort()

# im = Image.open(os.path.join(video_path, '000001.jpg'))
# ims=np.array(im)
# print(ims.shape)

# im = Image.open("F:/video_action_recognition/flow_output/mouse1")
# im.show()
# img = np.array(im)      # image类 转 numpy
# img = img[:,:,0]        #第1通道
# im=Image.fromarray(img) # numpy 转 image类
# im.show()
# image_raw_data = tf.gfile.FastGFile(os.path.join(video_path, '000001.jpg'), 'rb').read()
# img_data = tf.image.decode_jpeg(image_raw_data)
# print(img_data.shape)
# with h5py.File("flow.hdf5", 'r') as f:  # 读取的时候是‘r’
#   print(f.keys())
#   flow = f.get("a_data")[:]
#   print(flow.shape)
#   # print(flow[1,1,1,0])
#   for i,flow3 in enumerate(flow) :
#     sum=0
#     for cow,flow2 in enumerate(flow3):
#       for column,flow1 in enumerate(flow2):
#         sum+=((flow[i,cow,column,0]**2+flow[i,cow,column,1]**2)**0.5)
#     print(sum)

video_paths = "F:/video_action_recognition/frame_output/mouse1"
print(sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=os.path.getmtime))
