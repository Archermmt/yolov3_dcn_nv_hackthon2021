from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
import time
import cv2
import json

import argparse
parser=argparse.ArgumentParser(description='Coco eval')
#parser.add_argument('--trt_path',default="../../network/engine_code/yolov3_dcn_fp32/yolov3_dcn_1.trt",type=str,help='engine file path')
parser.add_argument('--trt_path',default="",type=str,help='tensorrt engine file')
parser.add_argument('--paddle_path',default="/usr/local/quake/datas/models/yolov3_dcn_paddle",type=str,help='paddle model file')
args,unknown=parser.parse_known_args()

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = "/data/tianhz/PaddleDetection-release-2.0-rc/dataset/coco"
dataType = "val2017"
annFile = os.path.join(dataDir, "annotations/instances_val2017.json")
gt_path = annFile
dt_path = "QuakeGenBoxes.json"

# 均值 方差
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
input_h = 608
input_w = 608

total_time = 0

coco = COCO(annFile)
#init engine/model
trt_engines,stream=None,None
paddle_executor=None
program,feed_names,target_vars=None,None,None

if args.trt_path:
    from trt_utils import *
    trt_engines,stream=init_trt_engine(args.trt_path)
elif args.paddle_path:
    import paddle
    import paddle.fluid as fluid
    paddle_executor=fluid.Executor(fluid.CUDAPlace(0))
    if hasattr(paddle, 'enable_static'):
      paddle.enable_static()
    program,feed_names,target_vars=fluid.io.load_inference_model(dirname=args.paddle_path,executor=paddle_executor)

# 获取annids
#annids = coco.getAnnIds()
#info = coco.loadAnns(annids[0])

# 获取类别数目 80类
# cat_ids = coco.getCatIds()

# 获取类别详细信息
# 数据结构[{'id':类别ID, 'name':类别名}]
#cats = coco.loadCats(cat_ids)

# 获取图像ID
imgs_ids = coco.getImgIds()
total_imgs = len(imgs_ids)

ret_list = list()

# 获取图像详细信息
# 数据结构[{'file_name': 文件名, 'coco_url': 文件url, 'height': 图片高, 'width': 图片宽, 'id': 图片ID}]
for img_idx,elem in enumerate(imgs_ids):
    img = coco.loadImgs([elem])[0]
    file_name = os.path.join(dataDir, 'val2017', img['file_name'])

    # load图像,利用opencv
    pic = cv2.imread(file_name, 1)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic_size=np.array(pic.shape[:2],dtype='int32')
    pic = cv2.resize(pic, (input_w, input_h))
    pic = (pic - mean)/std
    # c x h x w
    pic = pic.transpose(2, 0, 1).astype('float32')

    # forward
    start = time.time()

    input_datas={
        'im_id':np.array([elem],dtype='int64')[np.newaxis,:],
        'im_size':pic_size[np.newaxis,:],
        'image':pic[np.newaxis,:]}
    if trt_engines:
        #infeence_QuakeRT()
        results,nms_nums,elem_id=trt_infer(stream,trt_engines,input_datas['im_id'],input_datas['im_size'],input_datas['image'])
    elif paddle_executor:
        #inferenc_paddle()
        results,image_id=paddle_executor.run(program=program,feed=input_datas,fetch_list=target_vars,return_numpy=False)

    end = time.time()

    total_time += (end - start)
    # save result
    # example
    # [{"image_id":42, "category_id":18,"bbox":[258.15,41,29,348.26,243.78],"score":0.236}]
    if trt_engines:
        for cls_id,num in enumerate(nms_nums[0]):
            for box_id in range(num):
                box_info = dict()
                box_info['image_id'] = elem
                box_info['category_id'] = cls_id
                box_data=results[0][cls_id][box_id].tolist()
                #x1,y1,x2,y2
                box_info['bbox'] = [box_data[i] for i in range(4)]
                box_info['score'] = box_data[4]
                print("{}/{} th box info {}".format(img_idx,len(imgs_ids),box_info))
                ret_list.append(box_info)
    elif paddle_executor:
        for r in np.array(results):
            box_data=r.tolist()
            if box_data[0]==-1:
                continue
            box_info = dict()
            box_info['image_id'] = elem
            box_info['category_id'] = int(r[0])
            #x1,y1,x2,y2
            bbox = [box_data[i] for i in range(2,6)]
            bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            box_info['bbox']=bbox
            box_info['score'] = box_data[1]
            print("{}/{} th box info {}".format(img_idx,len(imgs_ids),box_info))
            ret_list.append(box_info)
    #if img_idx>=1000:
    #    break

# 保存文件
with open(dt_path, "w") as f:
    json.dump(ret_list, f)

print("time: ", total_time / total_imgs, "s")

# coco精度
cocoGt = COCO(gt_path)
cocoDt = cocoGt.loadRes(dt_path)
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


