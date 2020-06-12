# import os
#
#
# class DataSet():
#   def __init__(self, num_of_snip=1, opt_flow_len=10, image_shape=(224, 224), original_image_shape=(341, 256),
#                class_limit=None):
#     """Constructor.
#     opt_flow_len = (int) the number of optical flow frames to consider
#     class_limit = (int) number of classes to limit the data to.
#         None = no limit.
#     """
#     self.opt_flow_len = opt_flow_len
#     self.num_of_snip = num_of_snip
#     self.class_limit = class_limit
#     self.image_shape = image_shape
#     self.original_image_shape = original_image_shape
#     # self.opt_flow_path = os.path.join('/data', 'opt_flow')
#     self.opt_flow_path = os.path.join('F:/', 'test')
#
#     # Get the data.
#     self.data_list = self.get_data_list()
#
#     # Get the classes.
#     self.classes = self.get_classes()
#
#     # Now do some minor data cleaning
#     self.data_list = self.clean_data_list()
