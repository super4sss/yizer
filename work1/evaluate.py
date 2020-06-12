from keras.engine.saving import load_model

from work1.DataSet import DataSet

dataset = DataSet
flow_paths = "F:/video_action_recognition/flow_output/test1"
flow_hdf5='mouse3.hdf5'
genertor = DataSet.generate(dataset,30,flow_paths,flow_hdf5)

model = load_model('model/weights.01-2.1912-0.8333.hdf5')
loss, accuracy = model.evaluate_generator(genertor,steps=11,verbose=1)
print('\ntest loss', loss)
print('accuracy', accuracy)

# 保存参数，载入参数
# model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')
# 保存网络结构，载入网络结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)

print(json_string)
