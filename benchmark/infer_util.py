from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
np.set_printoptions(suppress=True)
import os
import time
import cv2
import json

class infer_base(object):
    def __init__(self, model_dir, batch, run_benchmark, param_type, warm_up, debug):
        self.model_dir = model_dir
        self.run_benchmark = run_benchmark
        self.warm_up = warm_up
        self.debug = debug
        self.param_type = param_type

        self.dataDir = "./dataset/coco"
        self.annFile = os.path.join(self.dataDir, "annotations/instances_val2017.json")
        self.gt_path = self.annFile
        self.dt_path = "QuakeGenBoxes.json"
        self.coco = COCO(self.annFile)
        self.cat_ids = self.coco.getCatIds()
        self.imgs_ids = self.coco.getImgIds()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.batch_size = batch
        self.input_h = 608
        self.input_w = 608
        self.total_time = 0

        self.print_num = 504

    def load_data(self, index):
        pic_list = list()
        info_list = list()
        for i in range(index, index + self.batch_size):
            elem = self.imgs_ids[i]
            img = self.coco.loadImgs([elem])[0]
            file_name = os.path.join(self.dataDir, 'val2017', img['file_name'])
            pic = cv2.imread(file_name, 1)
            pic_h = pic.shape[0]
            pic_w = pic.shape[1]

            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            scale_x = float(self.input_w) / float(pic_w)
            scale_y = float(self.input_h) / float(pic_h)
            pic = cv2.resize(
                pic,
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=2)
            pic = (pic - self.mean) / self.std
            pic = pic.transpose((2, 0, 1)).copy()
            pic_list.append(pic)
            info_list.append([pic_h, pic_w])

        return np.array(pic_list).astype('float32'), np.array(info_list).astype('int32')

    def accuracy(self):
        cocoGt = COCO(self.gt_path)
        cocoDt = cocoGt.loadRes(self.dt_path)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def run(self):
        pass

class infer_paddle(infer_base):
    def __init__(self, model_dir, batch, run_benchmark, param_type, warm_up, debug):
        super().__init__(model_dir, batch, run_benchmark, param_type, warm_up, debug)

        self.predictor = None
        self.model_dir = model_dir

        self.init_model()

    def init_model(self):
        from paddle.inference import Config
        from paddle.inference import PrecisionType
        from paddle.inference import create_predictor

        precision_mode = PrecisionType.Float32
        use_calib_mode = False

        if self.param_type == "fp16":
            precision_mode = PrecisionType.Half
        elif self.param_type == "int8":
            precision_mode = PrecisionType.Int8
            use_calib_mode = True

        mode_path = os.path.join(self.model_dir,"__model__")
        param_path = os.path.join(self.model_dir,"__params__")
        config = Config(mode_path, param_path)
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
        size = (self.batch_size * 101) << 20
        config.enable_tensorrt_engine(
            workspace_size= size,
            max_batch_size=self.batch_size,
            min_subgraph_size=3,
            precision_mode=precision_mode,
            use_static=False,
            use_calib_mode=use_calib_mode)
        if not self.debug:
            config.disable_glog_info()
        else:
            config.enable_profile()

        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        config.enable_mkldnn()
        #exit(1)
        self.predictor = create_predictor(config)


    def run(self):
        input_names = self.predictor.get_input_names()
        input_img = self.predictor.get_input_handle(input_names[0])
        input_info = self.predictor.get_input_handle(input_names[1])
        total_time = 0
        total_imgs = 0
        ret_list = list()
        print_num = self.print_num

        if self.run_benchmark:
            # warp up
            if self.warm_up != 0:
                pics, infos = self.load_data(0)
                input_img.copy_from_cpu(pics)
                input_info.copy_from_cpu(infos)

                for index in range(0, self.warm_up):
                    self.predictor.run()

            pics_list = list()
            info_list = list()
            im_list = list()
            print("[INFO] Paddle loading data.......")
            for index in range(0, len(self.imgs_ids), self.batch_size):
                if (index + self.batch_size) > 5000:
                    break
                pics, infos = self.load_data(index)
                pics_list.append(pics)
                info_list.append(infos)
                im_id_list = list()
                for i in range(index, index + self.batch_size):
                    im_id_list.append(self.imgs_ids[i])
                im_id = np.array(im_id_list, dtype='int32')[np.newaxis, :]
                im_list.append(im_id)
            print("[INFO] Paddle run inference.....")

            for i in range(0, len(pics_list)):
                input_img.copy_from_cpu(pics_list[i])
                input_info.copy_from_cpu(info_list[i])

                start = time.time()
                self.predictor.run()
                end = time.time()
                total_time += (end - start)
                total_imgs += 1

                if (i * self.batch_size) > print_num:
                    print("The average time for ", print_num, " images is ", (total_time / total_imgs) * 1000 ,"ms")
                    print_num += self.print_num

            print("batch_size = ", self.batch_size ,"cost time: ", (total_time / total_imgs) * 1000, "ms")

        else:
            for index in range(0, len(self.imgs_ids)):
                pics, infos = self.load_data(index)
                input_img.copy_from_cpu(pics)
                input_info.copy_from_cpu(infos)

                start = time.time()
                self.predictor.run()
                end = time.time()
                total_time += (end - start)
                total_imgs += 1


                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_handle(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()

                # ignore
                if len(np_boxes[0]) != 6:
                    print("No object detected in the images: ", index)
                    print(np_boxes.shape)
                    continue

                for boxx in np_boxes:
                    box_info = dict()
                    box_info['image_id'] = int(self.imgs_ids[index])
                    box_info['category_id'] = self.cat_ids[int(boxx[0])]
                    box_info['bbox'] = [float(boxx[2]), float(boxx[3]), float(boxx[4] - boxx[2]), float(boxx[5] - boxx[3])]
                    box_info['score'] = float(boxx[1])
                    ret_list.append(box_info)

            print("time: ", (total_time / total_imgs) * 1000, "ms")
            #save file
            with open(self.dt_path, "w") as f:
                json.dump(ret_list, f)

            self.accuracy()





class infer_quakert(infer_base):
    def __init__(self, model_dir, batch, run_benchmark, param_type, warm_up, debug):
        self.trt_engines = None
        self.stream = None

        super().__init__(model_dir, batch, run_benchmark, param_type, warm_up, debug)
        self.init_model()

    def init_model(self):
        from trt_utils import init_trt_engine

        build_path = None

        if self.param_type == "fp32":
            build_path = "../network/engine_code/yolov3_dcn_fp32"
        elif self.param_type == "fp16":
            build_path = "../network/engine_code/yolov3_dcn_fp16"
        elif self.param_type == "int8":
            build_path = "../network/engine_code/yolov3_dcn_int8"
        else:
            print("[ERROR] param_tyep not supprot")
            exit(1)

        model_path = os.path.join(build_path, "yolov3_dcn_1.trt")
        if os.path.exists(model_path):
            os.remove(model_path)

        # switch path
        cur_path = os.getcwd()
        os.chdir(build_path)
        ret = os.system("make && ./yolov3_dcn_1 " + str(self.batch_size))
        os.chdir(cur_path)

        if ret != 0 :
            print("[build trt model failed.]")
            exit(1)

        if not os.path.exists(model_path):
            print("[ERROR] model file no exist.")
            exit(1)

        self.trt_engines, self.stream = init_trt_engine(model_path)
        print("[INFO] init trt model success.......")

    def run(self):
        print("[INFO] start run trt model..........")
        from trt_utils import trt_infer
        ret_list = list()
        total_time = 0
        total_imgs = 0
        print_num = self.print_num

        if self.run_benchmark:
            # warp up
            if self.warm_up != 0:
               pass

            pics_list = list()
            info_list = list()
            im_list = list()
            print("[INFO] trt loading data.......")
            for index in range(0, len(self.imgs_ids), self.batch_size):
                if (index + self.batch_size) > 5000:
                    break
                pics, infos = self.load_data(index)
                pics_list.append(pics)
                info_list.append(infos)
                im_id_list = list()
                for i in range(index, index + self.batch_size):
                    im_id_list.append(self.imgs_ids[i])
                im_id = np.array(im_id_list, dtype='int32')[np.newaxis, :]
                im_list.append(im_id)

            print("[INFO] trt run inference.....")
            for i in range(0,len(pics_list)):
                im_id = im_list[i]
                infos = info_list[i]
                pics = pics_list[i]
                start = time.time()
                results, nms_nums, elem_id = trt_infer(self.stream, self.trt_engines, im_id, infos, pics)
                end = time.time()
                total_time += (end - start)
                total_imgs += 1
                if (i * self.batch_size) > print_num:
                    print("[TRT] The average time for ", print_num, " images is ", (total_time / total_imgs) * 1000 ,"ms")
                    print_num += self.print_num

            print("batch_size = ", self.batch_size, "cost time: ", (total_time / total_imgs) * 1000, "ms")

        else:
            self.batch_size = 1
            for index in range(0, len(self.imgs_ids)):
                pics, infos = self.load_data(index)
                im_id_list = list()
                for i in range(index, index + self.batch_size):
                    im_id_list.append(self.imgs_ids[i])
                im_id = np.array(im_id_list, dtype='int32')[np.newaxis, :]
                start = time.time()
                results, nms_nums, elem_id = trt_infer(self.stream, self.trt_engines, im_id, infos, pics)
                end = time.time()
                total_time += (end - start)
                total_imgs += 1

                for cls_id, num in enumerate(nms_nums[0]):
                    for box_id in range(num):
                        box_info = dict()
                        box_info['image_id'] = int(self.imgs_ids[index])
                        box_info['category_id'] = self.cat_ids[cls_id]
                        box_data = results[0][cls_id][box_id].tolist()
                        # x1,y1,x2,y2
                        #box_info['bbox'] = [box_data[i] for i in range(4)]
                        box_info['bbox'] = [float(box_data[0]), float(box_data[1]), float(box_data[2] - box_data[0]),float(box_data[3] - box_data[1])]
                        box_info['score'] = box_data[4]
                        #print(box_info)
                        #print("{}/{} th box info {}".format(img_idx, len(imgs_ids), box_info))
                        ret_list.append(box_info)

            print("time: ", (total_time / total_imgs) * 1000, "ms")
            # save file
            with open(self.dt_path, "w") as f:
                json.dump(ret_list, f)

            self.accuracy()