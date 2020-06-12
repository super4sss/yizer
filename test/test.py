import numpy
import tensorflow as tf

# hello = tf.constant('hello,tf')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))
# print(tf.is_tensor(sess))
# print(hello.dtype)
# # hello.device
# s = tf.constant("1",dtype="string")
# s.numpy
# a=bool(s)
# b= numpy.ones(2,3)
# print(b)
# x = tf.constant([1,2.,3.3])
# a = tf.constant([1.2])
# a=tf.ones([10,9])
# b=tf.zeros_like(a)
# b=tf.fill([9],2)

# b=tf.random.normal([8,9],mean=50,stddev=1)
# b=tf.random.uniform([3,4], minval=0, maxval=10, dtype=tf.int32)
# b=tf.range(10)
# b=tf.range(1,10,2)
# print(b)

# out = tf.random.uniform([4,10]) #随机模拟网络输出
# y = tf.constant([2,3,2,0]) # 随机构造样本真实标签
# y = tf.one_hot(y, depth=10) # one-hot 编码
# loss = tf.keras.losses.mse(y, out) # 计算每个样本的 MSE
# loss = tf.reduce_mean(loss) # 平均 MSE
# print(loss)


# y = tf.constant([0,1,2,3]) # 数字编码
# y = tf.one_hot(y, depth=5) # one-hot 编码
# s = tf.reshape(y,[1,20])
# print(s)

# a = tf.constant([0, 1, 2, 3, 4, 5,6,7,8,9])
#
# print(a[::-2])
#
# x = tf.random.normal([4,32,32,3])
# print(x)
# print(x.ndim)
# 维度变换
# x=tf.reshape(x,[4,-1])
# x=tf.reshape(x,[4,32,-1])
# print(x)
# print(x[:,0:28:2,0:28:2,:])
# 维度增删
# x=tf.expand_dims(x, -5)
# print(x.shape)
# 交换维度
# x = tf.random.normal([2,32,32,3])
# x=tf.transpose(x,perm=[0,3,1,2])
# print(x.shape)
from keras_applications.densenet import layers

from tensorflow.keras import Sequential
network = Sequential([  # 网络容器
layers.Conv2D(6,kernel_size=3,strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
layers.MaxPooling2D(pool_size=2,strides=2),  # 高宽各减半的池化层
layers.ReLU(),  # 激活函数
layers.Conv2D(16,kernel_size=3,strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
layers.MaxPooling2D(pool_size=2,strides=2),  # 高宽各减半的池化层
layers.ReLU(),  # 激活函数
layers.Flatten(),  # 打平层，方便全连接层处理
layers.Dense(120, activation='relu'),  # 全连接层，120 个节点
layers.Dense(84, activation='relu'),  # 全连接层，84 节点
layers.Dense(10)  # 全连接层，10 个节点
])
# build 一次网络模型，给输入 X 的形状，其中 4 为随意给的 batchsz
network.build(input_shape=(4, 28, 28, 1))
# 统计网络信息
network.summary()

# # 记录预测正确的数量，总样本数量
# correct, total = 0,0
# for x,y in db_test: # 遍历所有训练集样本
# # 插入通道维度，=>[b,28,28,1]
# x = tf.expand_dims(x,axis=3) # 前向计算，获得 10 类别的预测分布，[b, 784] => [b, 10]
# out = network(x)
# # 真实的流程时先经过 softmax，再 argmax
# # 但是由于 softmax 不改变元素的大小相对关系，故省去
# pred = tf.argmax(out, axis=-1) y = tf.cast(y, tf.int64)
# # 统计预测正确数量
# correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.floa
# t32)))
# # 统计预测样本总数
# total += x.shape[0] # 计算准确率
# print('test acc:', correct/total)
