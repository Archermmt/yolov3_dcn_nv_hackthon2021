import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import pylab
import os
import time
import cv2
import json
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor

images_dir = './dataset/images'
# 均值 方差
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
batch_size = 8
input_h = 608
input_w = 608

mode_path = "./model/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/__model__"
param_path = "./model/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco/__params__"
config = Config(mode_path, param_path)
config.enable_use_gpu(100, 0)
config.switch_ir_optim(True)
config.enable_tensorrt_engine(
    workspace_size=1 << 10,
    max_batch_size=batch_size,
    min_subgraph_size=3,
    precision_mode=PrecisionType.Int8,
    use_static=False,
    use_calib_mode=True)

config.enable_memory_optim()
config.switch_use_feed_fetch_ops(False)
config.enable_mkldnn()
#config.enable_profile()

predictor = create_predictor(config)
input_names = predictor.get_input_names()
input_img = predictor.get_input_handle(input_names[0])
input_info = predictor.get_input_handle(input_names[1])

for name in os.listdir(images_dir):
    pic_list = list()
    info_list = list()

    file_name = os.path.join(images_dir, name)
    pic = cv2.imread(file_name, 1)

    pic_h = pic.shape[0]
    pic_w = pic.shape[1]

    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    scale_x = float(input_w) / float(pic_w)
    scale_y = float(input_h) / float(pic_h)
    pic = cv2.resize(
        pic,
        None,
        None,
        fx=scale_x,
        fy=scale_y,
        interpolation=2)
    pic = (pic - mean) / std
    pic = pic.transpose((2, 0, 1)).copy()
    pic_list.append(pic)
    info_list.append([pic_h, pic_w])

    pics = np.array(pic_list)
    input_img.copy_from_cpu(pics.astype('float32'))
    input_info.copy_from_cpu(np.array(info_list).astype('int32'))
    predictor.run()
