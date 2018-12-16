# 智创工程AI挑战赛暨人工智能与轨道安全数据竞赛top10代码

需要下载数据并在root下新建为以下格式:

```python
# 训练及测试数据
├── DATA
│   ├── Amend_VOC2007  # 用于微调模型的数据集,VOC 2007 格式
│   ├── test  # 测试集
│   └── VOC2007  # 训练集
```

**官方数据地址**: http://hydinin.com:8086/phm/article11.html

**我们制作的数据集**:  

## 1. Introduction

​	我们最终取得了第10名的成绩，使用的是`YOLO V3`作为比赛的框架。大概是第一次参加比赛的缘故，归结原因:

- 萌新不知道如何正确调参
- 也许模型选择可能有问题，我们也尝试了何凯明大神最新的`Retinanet`，在初赛的时候准确率只能到95%左右
- 算法细节没有深入修改

所以最终用的测试权重是11.25号第一次跑出来的权重，我们的算法应该还有改进的空间。但最终`YOLO V3`我们做到的最高准确率也就在 98.819，一直提高不上去。

![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181130154358.png)

根据观察训练集，我们发现裂纹：

- 大多呈现连续狭长型，少部分为细小裂纹，
- 部分细小裂纹和一些背景即使肉眼也难以区分

制作好训练集后做了一个简单的模型预测，观察预测结果我们发现以下几个特点：

- 当预测出候选框超过3个以上时候，模型几乎不会误判。主要误判原因在于
  1. 头发丝被当做裂纹
  2. 很多碎小的石头块被当做裂纹

  ![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181216130520.png)

- 候选框在2-3个时候，模型误判概率也并不大。主要误判原因在于：
  1. 石头被当做裂纹
  2. 图片下方有些干扰背景

  ![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181216130754.png)

- 当候选框在1个时候，模型误判概率很大。主要误判原因在于：
  1. 无法正确区分前景背景
  2. 裂纹和一些几何交汇面无法区分
  3. 存在斑点和裂纹难以区分现象
  4. 误判，有些凹痕被当做裂纹
  5. 漏判，有些裂纹由于和背景融合一起，模型无法识别出

  ![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181216130848.png)

- 当候选框在0个时候，模型存在一些漏判的情况

![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181216131003.png)

###  技术路线简介

因此我们主要依靠 **预测出裂纹数量** 以及 **候选框多级阀值**来作为裂纹的判别依据来预测裂纹：

![](https://raw.githubusercontent.com/SHU-FLYMAN/image_picgo/master/20181201175233.png)

## 2. 主要代码文件

```sh
# 主要文件
├── kmeans.py
├── yolo.py  # 
├── voc_annotation.py  # 数据集转换,从voc到coco
├── train.py  # 训练文件
├── test.py  # 测试文件
├── amend.py  # 修正文件
├── final_result.txt  # 最终预测结果

# 训练及测试数据
├── DATA
│   ├── Amend_VOC2007  # 用于微调模型的数据集,VOC 2007 格式
│   ├── test  # 测试集
│   └── VOC2007  # 训练集

├── logs  # 日志文件夹

├── model_data  # 存放训练集信息
│   ├── train.txt  # 训练集
│   ├── 2007_train.txt
│   ├── tiny_yolo_anchors.txt
│   ├── voc_classes.txt  # 用于修改voc类别
│   ├── weights_pretrained.h5  # 预训练的模型权重,用于微调
│   └── yolo_anchors.txt


├── Results  # 结果文件夹
│   └── test
│       ├── amend
│       │   ├── amend.txt  # 修正后的结果
│       │   ├── crack
│       │   └── no_crack
│       └── predict
│           ├── crack
│           ├── need_amend.txt
│           ├── no_crack
│           └── result.txt  # 第一次预测的结果

├── Weights  # 存放权重
│   ├── amend_weights.h5
│   └── test_weights.h5

├── yolo3  # yolo3网络主体
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
```

**使用顺序**

```sh
1. 将训练数据以VOC2007格式放置到DATA/VOC2007文件夹
2. 运行train.py文件，其会调用voc_annotation.py自动将数据从voc2007转换到coco格式
3. 运行test.py，利用预测网络预测结果，将有很大把握是裂纹的图片归类为裂纹，没有把握的图片存放在need——amend.txt 文件中，传递给校正网络校正
4. 再次运行train.py,这次修改参数训练含有空白图片的校正数据集
5. 利用amend.py 校正网络结果，最终结果保存在final_result.txt中
```

###2.1 训练

```sh
python train.py  # 开启训练
```

```sh
usage: train.py [-h] [--usage {train,amend}] [-v VALID_RATE]
                [--load_model LOAD_MODEL] [--weight_path WEIGHT_PATH]
                [--epoch_1 EPOCH_1] [--epoch_2 EPOCH_2]
                [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --usage {train,amend}
                        训练/矫正模型权重, default=train
  -v VALID_RATE, --valid_rate VALID_RATE
                        验证机划分比重, default=0.1
  --load_model LOAD_MODEL
                        是否加载预训练权重,if
                        load_model=True, 如果为True,你需要指定模型权重weight_path,如果不指定,默认的weight_path=model_data/weights_pretrained.h5
  --weight_path WEIGHT_PATH
                        预训练权重路径,只有当 `--load_model True`时候才有效
  --epoch_1 EPOCH_1     冻结卷积层,只训练全连接层的训练轮数
  --epoch_2 EPOCH_2     训练整个神经网络的训练轮数
  --batch_size BATCH_SIZE
                        指定Batchsize,默认为4
```

### 2.2 测试

```sh
python test.py  # 开启测试
```

```sh
usage: test.py [-h] [--model_path MODEL_PATH] [--min_threshold MIN_THRESHOLD]
               [--all_threshold ALL_THRESHOLD] [--two_threshold TWO_THRESHOLD]
               [--one_threshold ONE_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        the model weight of testing
  --min_threshold MIN_THRESHOLD
                        the test min socre used to filterate boxes,default=0.2
  --all_threshold ALL_THRESHOLD
                        the threshold when the model predoct at least 4
                        anchors
  --two_threshold TWO_THRESHOLD
                        the test threshold used to filterate the boxes which
                        be predicted two boxes,default=0.35
  --one_threshold ONE_THRESHOLD
                        the test threshold used to filterate the boxes which
                        be predicted only one box,default=0.4
```

### 2.3 矫正结果

```sh
python amend.py
```

```sh
usage: amend.py [-h] [--model_path MODEL_PATH] [--min_threshold MIN_THRESHOLD]
                [--max_threshold MAX_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        the model weight of amended model,
                        default=Weights/test_weights.h5
  --min_threshold MIN_THRESHOLD
                        the amended min socre used to filterate
                        boxes,default=0.3
  --max_threshold MAX_THRESHOLD
                        the amended max socre used to filterate
                        boxes,default=0.7
```



