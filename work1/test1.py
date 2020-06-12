import glob
import random
import os
# import tensorflow as tf # 导入 TF 库
# from tensorflow import keras # 导入 TF 子库
# from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库
# (x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载数据集
# x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1 # 转换为张量，缩放到-1~1
# y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为张量
# y = tf.one_hot(y, depth=10) # one-hot 编码
# print(x.shape, y.shape)

# train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
# train_dataset = train_dataset.batch(512) # 批量训练

# network =  Sequential([  # 网络容器
# layers.Conv2D(6,kernel_size=3,strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
# layers.MaxPooling2D(pool_size=2,strides=2),  # 高宽各减半的池化层
# layers.ReLU(),  # 激活函数
# layers.Conv2D(16,kernel_size=3,strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
# layers.MaxPooling2D(pool_size=2,strides=2),  # 高宽各减半的池化层
# layers.ReLU(),  # 激活函数
# layers.Flatten(),  # 打平层，方便全连接层处理
# layers.Dense(120, activation='relu'),  # 全连接层，120 个节点
# layers.Dense(84, activation='relu'),  # 全连接层，84 节点
# layers.Dense(10)  # 全连接层，10 个节点
# ])
# # build 一次网络模型，给输入 X 的形状，其中 4 为随意给的 batchsz
# network.build(input_shape=(4, 28, 28, 1))
# # 统计网络信息
# network.summary()

# a = [1, 1, 2, 4, 5, ]
# # for i in a:
# #   print(random.choice(a))
# print(a[3:9])

# with h5py.File("flow5.hdf5", 'w') as f:
#   f.create_dataset("a_data", data=[1,24,5,7,5], compression="gzip", compression_opts=5)
import h5py
import  numpy as np
# with h5py.File('flow1.hdf5', 'r') as hf:
#   data = hf['a_data'][:]
# data = np.array(data)
# data = np.delete(data, 0, axis=0)
# data = data[1:361]
# print(data.shape)

# flags = []
# flags = np.array(flags)
# flow_paths = "F:/video_action_recognition/flow_output/test1"
# frames = sorted(glob.glob(os.path.join(flow_paths, '*.jpg')), key=os.path.getmtime)
# frames = glob(os.path.join(flow_paths, '*.jpg'))
# for i, path in enumerate(frames):
#   flag = path.split('.')
#   # flags = np.append(flags, int(flag[1]))
#   print(flag[0])
# print(frames[99:88])
with h5py.File('mouse1.hdf5', 'r') as hf:
  data1 = hf['mouse1_data'][:]
with h5py.File('mouse2.hdf5', 'r') as hf:
  data2 = hf['mouse2_data'][:]
data1 = np.delete(data1, 0, axis=0).tolist()
data2 = np.delete(data2, 0, axis=0).tolist()
data1.extend(data2)
print(data1)

# with h5py.File('mouse2.hdf5', 'r') as hf:
#   data = hf['mouse2_data'][:]
# data = np.array(data)
# data = np.delete(data, 0, axis=0)
# # data = data[1:361]
# print(data)
