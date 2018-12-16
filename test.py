import os
import shutil
import argparse
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from yolo import YOLO
from PIL import Image


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


K.tensorflow_backend.set_session(get_session())


class PredictImages(object):
    """预测裂纹"""
    def __init__(self, model_path, min_threshold=0.2, all_threshold=0.3, two_threshold=0.35, one_threshold=0.4):
        """根据裂纹分布特点，当预测出框越多时，该图片含有裂纹的概率越大。之后利用多值过滤scores预测裂纹

        Args:
            model_path: 模型权重路径
            min_threshold: box的最低阀值，只有预测出分数大于这个值，框才会被算入
            all_threshold: 预测出大于3个框时的阀值
            two_threshold: 预测出两到三个框的阀值
            one_threshold: 预测出一个框的阀值
        Examples:
            predict = PredictImages(model_path='Weights/test_weights.h5',
                            min_threshold=0.2,
                            all_threshold=0.3,
                            two_threshold=0.35,
                            one_threshold=0.4)
            predict.predict_images()
        """
        self.recheck_images = []
        self.image_dir = 'DATA/test'
        self.outdir = 'Results/test'
        self.min_threshold = min_threshold
        self.all_threshold = all_threshold
        self.two_threshold = two_threshold
        self.one_threshold = one_threshold
        self.model_path = model_path
        self.yolo = YOLO(self.model_path, min_threshold)
        self.result_folder_path = self._creat_folder()

    @staticmethod
    def _dele_creat_folder(folder_path):
        """清空之前内容，创建新文件夹"""
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    def _creat_folder(self):
        result_folder_path = os.path.join(self.outdir, 'predict')
        for i in ['crack', 'no_crack']:
            self._dele_creat_folder(os.path.join(result_folder_path, i))
        return result_folder_path

    def _detect_one_image(self, img, boxes, scores, filename):
        """预测逻辑"""
        len_boxes = len(boxes)
        if len_boxes == 0:
            img.save(os.path.join(self.result_folder_path, 'no_crack', filename))
            return 0
        elif len_boxes > 3:
            if max(scores) < self.all_threshold:
                img.save(os.path.join(self.result_folder_path, 'no_crack', filename))
                self.recheck_images.append(filename)
                return 0
            else:
                img.save(os.path.join(self.result_folder_path, 'crack', filename))
                return 1
        elif 2 <= len_boxes <= 3:
            if max(scores) < self.two_threshold:
                img.save(os.path.join(self.result_folder_path, 'no_crack', filename))
                self.recheck_images.append(filename)
                return 0
            else:
                img.save(os.path.join(self.result_folder_path, 'crack', filename))
                return 1
        else:
            if scores[0] > self.one_threshold:
                if boxes[0][0] < 250:
                    img.save(os.path.join(self.result_folder_path, 'no_crack', filename))
                    return 0
                else:
                    img.save(os.path.join(self.result_folder_path, 'crack', filename))
                    return 1
            else:
                img.save(os.path.join(self.result_folder_path, 'no_crack', filename))
                self.recheck_images.append(filename)
                return 0

    def predict_images(self):
        with open(os.path.join(self.result_folder_path, 'result.txt'), 'w') as f:
            for root, _, filenames in os.walk(self.image_dir):
                for filename in tqdm(sorted(filenames, key=lambda x: int(x.split('.')[0])), desc='开始预测测试集'):
                    filepath = os.path.join(self.image_dir, filename)
                    img = Image.open(filepath)
                    img, boxes, scores = self.yolo.detect_image(img)
                    result = self._detect_one_image(img, boxes, scores, filename)
                    f.write('{} {}\n'.format(filename, result))
        with open(os.path.join(self.result_folder_path, 'need_amend.txt'), 'w') as w:
            for i in self.recheck_images:
                w.write('{}\n'.format(i))
        print("需要修正结果的图片总共{}张".format(len(self.recheck_images)))
        self.yolo.close_session()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Weights/test_weights.h5', help='the model weight of testing')
    parser.add_argument('--min_threshold', type=float, default=0.2,
                        help='the test min socre used to filterate boxes,default=0.2')
    parser.add_argument('--all_threshold', type=float, default=0.3,
                        help='the threshold when the model predoct at least 4 anchors')
    parser.add_argument('--two_threshold', type=float, default=0.35,
                        help='the test threshold used to filterate the boxes which be predicted two boxes,default=0.35')
    parser.add_argument('--one_threshold', type=float, default=0.4,
                        help='the test threshold used to filterate the boxes which be predicted only one box,default=0.4')
    args = parser.parse_args()
    predict = PredictImages(model_path=args.model_path,
                            min_threshold=args.min_threshold,
                            all_threshold=args.all_threshold,
                            two_threshold=args.two_threshold,
                            one_threshold=args.one_threshold)
    predict.predict_images()
