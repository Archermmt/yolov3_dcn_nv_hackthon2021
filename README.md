#yolov3_dcn_nv_hackthon2021
##简介
本开源内容为 英伟达-阿里云异构计算TensorRT 加速AI推理Hackathon 2021 比赛项目，利用TensorRT对模型的推理过程进行加速
##使用模型
选取yolov3模型作为原始模型，模型的结构可以直接从PaddleDetection上获取，非常方便。

PaddleDetection源码：https://github.com/PaddlePaddle/PaddleDetection

模型配置文件：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/configs/dcn/yolov3_r50vd_dcn_db_obj365_pretrained_coco.yml

##名词测试
| 名称 | 解释 |
| -----  | ----- |
| Paddle   |  百度的原始深度学习框架 |
| Paddle-trt   |  paddle集成了trt的推理库    |
| Quake   |  本次参赛的框架，集成了模型编译和推理（包含TensorRT推理）   |

##环境依赖
###硬件环境
注：本次比赛nvidia官方提供了开发用的基本docker镜像，镜像名称为nvcr.io/nvidia/tensorrt:21.02-py3

| 硬件类型 | 详细信息 |
| -----  | ----- |
| GPU   |  Tesla T4|
| CPU   |  Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz     |
| 内存   |  DDR4 30G     |
###软件环境
| 软件类型 | 版本 |
| ----- | ----- |
| 系统 | Ubuntu 7.5.0-3ubuntu1~18.04 |
| Gpu驱动 | 460.32.03 |
| Cuda | 11.2 |
| Cudnn | 8.2 |
| TensrRT | 7.2.2 |
| Python | 3.8.5 |
| gcc/g++ | 8.4.0 |
###python依赖包
| 库名称 | 版本 |
| ----- | ----- |
|Cython	|0.29.23|
|matplotlib|3.4.1|
|numpy	|1.19.5|
|nvidia-pyindex	|1.0.8|
|opencv-python|	4.5.1.48|
|paddlepaddle-gpu|	2.0.1(需要手动编译，否则无法支持TensorRT)|
|Pillow	|8.1.0 |
|polygraphy|	0.22.0|
|protobuf|	3.14.0|
|pycocotools|	2.0|
|pycuda|	2020.1|
|PyGObject|	3.36.0|
|python-dateutil|	2.8.1|
|pytools|	2021.1|
|PyYAML|	5.4.1|
|pyzmq|	22.0.3|
|requests|	2.25.1|
|scipy|	1.6.2|
|setuptools|	45.2.0|
|six|	1.15.0|
|tensorrt|	7.2.2.3|
|tqdm|	4.60.0|
|urllib3|	1.26.4|

###算子库环境
本项目中使用的模型优化代码和库需要进行单独编译才能使用，编译代码如下：
```shell
cd ops_lib && \
mkdir build && \
cd build && \
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/quake ../ && \
make -j40 install && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/quake/lib 
```

##目录结构介绍
|--benchmark  用于测试优化后模型性能的工具包\
    |--------dataset 用于测试的数据集\
    |--------images 量化数据集\
    |--------coco   进行精度和耗时测试的数据集\
    |--------model  存放已经导出的paddle模型\
    |--------create_paddle_quantization_data.py  生成paddle-trt的校准数据集\
    |--------infer.py  benchamark测试接口\
|--bug_codes  可以复现TensorRT使用过程中的bug的代码\
|--network    优化网络的核心代码\
    |--------visualize    可视化结构可以使用netron打开\
       |------------parse.prototxt   原始结构\
       |------------optimize.prototxt为优化后结构（可以使用原始框架进行inference）\
       |------------compile.prototxt为编译时结构（匹配tensorrt，但不能用原始框架进行inference）\
    |--------engine_code  构建engine的tensorrt代码\
|--ops_lib    比赛中使用到的算子库\
|--readme.md  本项目的详细介绍





##onnx转换
onnx转换需要安装paddle2onnx，本章节简单说明paddle的yolov3模型转onnx的操作,但该onnx模型无法通过trt的engine进行编译。
```shell
#安装onnx2paddle
pip instlal paddle2onnx
```
转换操作
```shell
# 该命令必须在本工程根目录下执行
paddle2onnx 
  --model_dir ./benchmark/model/yolov3/ \
  --model_filename __model__ \
  --params_filename __params__ \
  --save_file paddleDet_onnx/paddle_det.onnx \
  --opset_version 11
```

##精度对比
精度验证方案说明
>由于本次参赛选用的模型是yolov3检测模型，目前对于检测模型评估主要以mAP的大小来衡量模型的精度，故我们对于正确性的验证设计的方案为，比较同一个数据集的mAP，来验证优化后模型的正确性。选用的数据集为coco val2017数据集，其中包含5000张图片。

原始框架Paddle验证精度操作方法
```shell
# 首先需要下载PaddleDetection,并进入到paddle根目录
# 执行验证命令，整个过程会自动下载数据集，请保证磁盘空间足够大
python eval.py --config ./configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml 
```
Paddle-trt证精度操作方法
```shell
# 下载本工程代码，并进入到工程根目录
# 进入benchmark目录
cd ./进入benchmark

# 验证fp32模型精度
python infer.py -p Paddle -t fp32

# 验证fp16模型精度
python infer.py -p Paddle -t fp16

# 验证int8模型精度
python infer.py -p Paddle -t int8
```
Quake证精度操作方法
```shell
# 下载本工程代码，并进入到工程根目录
# 进入benchmark目录
cd ./进入benchmark

# 验证fp32模型精度
python infer.py -p Quake -t fp32

# 验证fp16模型精度
python infer.py -p Quake -t fp16

# 验证int8模型精度
python infer.py -p Quake -t int8
```

精度数据对比表（表中的数据为mAP结果）：

| 模型推理类型 | Paddle | Paddle-trt | Quake |
| ----- | ----- | ----- | ----- |
|Float32|	0.425|	0.409|	0.406|
|Float16|	无	|0.409|0.406|
|Int8	|无	|0.35	|0.396|


##性能测试
性能测试方案说明
>由于不同的图片推理耗时不同、硬件的状态每个时间点也不同，为了避免这些系统误差引起的性能耗时不准确，我们采用统计一个数据集的整体耗时，并且计算单张图片的平均耗时来衡量模型推理的性能，在进行正式测试前，需要预热100次，来降低系统误差，保证测试结果的有效性。会分别统计和对比FP32的、FP16和Int8的模型性能耗时。

说明：一致性测试方案采用自动编译 + 自动测试的，无需用户手动编译

###测试方法说明：
测试原始Paddle框架的性能，执行如下命令会测试完成后输出fps。\
注意：默认的eval只支持batch = 8 ,如果需要测试其他batch，需要手动修改配置文件。
文件路径：configs/dcn/yolov3_enhance_reader.yml
修改内容：EvalReader字段下的batch_size
```shell
# 首先需要下载PaddleDetection,并进入到paddle根目录
# 执行验证命令，整个过程会自动下载数据集，请保证磁盘空间足够大
python eval.py --config ./configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml 
```

测试Paddle-trt和Quake框架的推理性能使用benchmark目录下的infer.py脚本，命令参数表如下：

| 参数 | 描述 |
| ----- | ----- |
|-h, --help|	show this help message and exit|
|-p｜--platform|	Test platform type, support Paddle or QuakeRT, default is Paddle	|
|-m｜--model_dir |Test model path, default is ./model/yolov3	|
|-b｜--batch |Test batch size, default is 1	|
|-t｜--param_type |Inference data type, support fpr32 ｜ fp16 ｜ int8,default is fp32	|
|-r｜--run_benchmark |run benchmark test, default is False	|
|-w｜--warm_up |Warm-up times, default is 100	|
|-d｜--debug |open debug, default is False	|
例子：
```shell
# 测试Paddle-trt的fp32模型 batch = 8的性能
python infer.py -p Paddle -t fp32 -r 1 -b 8

# 测试Quake的fp32模型 batch = 8的性能
python infer.py -p Quake -t fp32 -r 1 -b 8

```
###fp32模型 Paddle原始框架 VS Quake
|平台| batch | Latency, ms | Throughput (1000/latency*batchsize)| Latency Speedup(original latency / TRT latency |
|---| ----- | ----- | ----- | ----- |
|Paddle|	1|	70.87|	14.11|		
|	|2	|87.3	|22.91|		
|	|4	|169.56|	23.59|		
|	|8	|325.73	|24.56|		
|	|16	|610.92	|26.19|		
|	|32	|1315.25	|24.33|		
|Quake	|1	|30.13	|33.19	|2.35|	
|	|2|	57.48|	34.79|	1.52|	
|	|4	|112.41	|35.58|	1.51|	
|	|8	|232.78|	34.36|	1.39|	
|	|16	|467.94|	34.19|	1.31|

###fp32模型 Paddle-trt VS Quake
|平台| batch | Latency, ms | Throughput (1000/latency*batchsize)| Latency Speedup(original latency / TRT latency |
|---| ----- | ----- | ----- | ----- |
|Paddle-TRT|	1|	30.93|	32.33|		
|	|2|	56.2|	35.59|		
|	|4|	113.1|  35.37|		
|	|8|	230.8|	34.66|		
|Quake|	1|	30.13|	33.19|	1.03	|
|	|2|	57.48	|34.79|	0.98|	
|	|4|	112.41	|35.58|	1.0|	
|	|8|	232.78	|34.36|	0.99|	
|	|16|	467.94|	34.19|

###fp16模型Paddle-trt VS Quake
|平台| batch | Latency, ms | Throughput (1000/latency*batchsize)| Latency Speedup(original latency / TRT latency |
|---| ----- | ----- | ----- | ----- |
|Paddle-TRT|	1|	13.32|	75.08|		
|	|2|	24.27|	82.41|		
|	|4|	46.67|	85.71|		
|	|8|	91.34|	87.58|		
|	|16|	191.77|	84.43|		
|Quake|	1|	10.12 |	98.81|	1.32|	
|	|2|	19.18|	104.28|	1.27|	
|	|4|	38.08|	105.04|	1.22|	
|	|8|	82.45|	97.03|	1.11|	
|	|16|	177.53|	90.13|	1.08|	

###int8模型Paddle-trt VS Quake
|平台| batch | Latency, ms | Throughput (1000/latency*batchsize)| Latency Speedup(original latency / TRT latency |
|---| ----- | ----- | ----- | ----- |
|Paddle-TRT	|1|	11.27|	88.73|		
|	|2|	21.11|	94.74		|
|	|4|	41.24|	97		|
|	|8|	78.62|	101.76	|	
|	|16|	164.85|	97.06|		
|Quake	|1|	7.71|	129.7|	1.46|	
|	|2|	14.85|	134.68|	1.42	|
|	|4|	30.82|	129.79|	1.33	|
|	|8|	69.41|	115.26|	1.14	|
|	|16|	143.91|	111.18|	1.15|
