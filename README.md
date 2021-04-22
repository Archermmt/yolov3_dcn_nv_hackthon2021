#yolov3_dcn_nv_hackthon2021
英伟达-阿里云异构计算TensorRT 加速AI推理Hackathon 2021 比赛项目
选取paddledetection 模型作为原始模型：
https://github.com/PaddlePaddle/PaddleDetection
配置使用：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/configs/dcn/yolov3_r50vd_dcn_db_obj365_pretrained_coco.yml

内容如下：
ops_lib：比赛中使用到的算子库

datas：使用到的数据（模型参数，测试数据等）

network：比赛用的网络构建代码以及相关网络信息
  visualize：可视化结构可以使用netron打开
    parse.prototxt为原始结构
    optimize.prototxt为优化后结构（可以使用原始框架进行inference）
    compile.prototxt为编译时结构（匹配tensorrt，但不能用原始框架进行inference）
  engine_code：构建engine的tensorrt代码

evaluate：精度测试使用的脚本

bugs：可以复现TensorRT使用过程中的bug的代码

测试引擎性能：
1.安装paddlepaddle：
pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

2.安装算子库：
cd ops_lib && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/quake ../ && make -j40 install

3.准备数据：
待添加

4.构建 && 测试性能
A.FP32模型：cd network/yolov3_dcn_fp32 && make && ./yolov3_dcn_1
B.FP16模型：cd network/yolov3_dcn_fp16 && make && ./yolov3_dcn_1
C.INT8模型：cd network/yolov3_dcn_int8 && make && ./yolov3_dcn_1
D.python 加载测试：cd evaluate/simple_test && python test_engine.py --engine=YOUR_ENGINE_PATH
  YOUR_ENGINE_PATH 为A，B，C过程中构建得到的.trt文件路径

5.测试精度:
cd evaluate && python coco_infer.py --trt_path=YOUR_TRT_PATH （测试tensorrt engine的精度）
  YOUR_TRT_PATH 为4.A，4.B，4.C过程中构建得到的.trt文件路径

cd evaluate && python coco_infer.py --paddle_path=YOUR_PADDLE_PATH （测试paddle model的精度）
  YOUR_PADDLE_PATH 为paddle模型路径，yolov3_dcn模型在准备数据完成后保存在/usr/local/quake/datas/models/yolov3_dcn_paddle
或:
cd evaluate && python coco_infer.py（测试paddle model的精度）


