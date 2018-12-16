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


class AmendImages(object):
    """根据预测裂纹的结果，对一些没有把握的裂纹再次预测"""
    def __init__(self, model_path='Weights/amend_weights.h5', min_threshold=0.3, max_threshold=0.7):
        self.recheck_images = []
        self.image_dir = 'DATA/test'
        self.outdir = 'Results/test'
        self.model_path = model_path
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.amend_folder_path = self._creat_folder()
        self.yolo = YOLO(self.model_path, 0.2)

    def _find_amend_images(self):
        with open('Results/test/predict/need_amend.txt') as f:
            for image in f.readlines():
                self.recheck_images.append(image.strip())

    @staticmethod
    def _dele_creat_folder(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    def _creat_folder(self):
        amend_folder_path = os.path.join(self.outdir, 'amend')
        for folder in ['crack', 'no_crack']:
            self._dele_creat_folder(os.path.join(amend_folder_path, folder))

        return amend_folder_path

    def _detect_one_image(self, img, boxes, scores, filename):
        len_box = len(boxes)
        if len_box == 0:
            img.save(os.path.join(self.amend_folder_path, 'no_crack', filename))
            return 0
        elif len_box > 1:
            if ((boxes[-1][0]) > 1000) or ((boxes[-2][0]) > 1000):
                img.save(os.path.join(self.amend_folder_path, 'no_crack', filename))
                return 0
            else:
                img.save(os.path.join(self.amend_folder_path, 'crack', filename))
                return 1
        else:
            if (scores[0]<self.min_threshold) or (scores[0]>self.max_threshold):
                img.save(os.path.join(self.amend_folder_path, 'no_crack', filename))
                return 0
            else:
                if boxes[0][0] < 250:
                    img.save(os.path.join(self.amend_folder_path, 'no_crack', filename))
                    return 0
                else:
                    img.save(os.path.join(self.amend_folder_path, 'crack', filename))
                    return 1


    def amend_images(self):
        self._find_amend_images()  # 将需要修正的图片读取到实例属性rechek_images
        with open(os.path.join(self.amend_folder_path, 'amend.txt'), 'w') as f:
            for filename in tqdm(self.recheck_images, desc="开始修正结果"):
                filepath = os.path.join(self.image_dir, filename)
                img = Image.open(filepath)
                img, boxes, scores = self.yolo.detect_image(img)
                result = self._detect_one_image(img, boxes, scores, filename)
                f.write('{} {}\n'.format(filename, result))
        self.yolo.close_session()
        self._summary_result()

    def _summary_result(self):
        """汇总结果"""
        predict_result = {}
        amend_result = {}
        with open('Results/test/predict/result.txt') as fr:
            for line in fr.readlines():
                line = line.strip()
                predict_result[line.split(' ')[0]] = line.split(' ')[-1]
        with open('Results/test/amend/amend.txt') as fa:
            for line in fa.readlines():
                line = line.strip()
                amend_result[line.split(' ')[0]] = line.split(' ')[-1]
        with open('final_result.txt', 'w') as f:
            for image in tqdm(predict_result.keys(), desc="汇总最终结果"):
                if image in amend_result.keys():
                    f.write('{} {}\n'.format(image, amend_result[image]))
                else:
                    f.write('{} {}\n'.format(image, predict_result[image]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Weights/amend_weights.h5',
                        help='the model weight of amended model, default=Weights/amend_weights.h5')
    parser.add_argument('--min_threshold', type=float, default=0.4,
                        help='the amended min socre used to filterate boxes,default=0.4')
    parser.add_argument('--max_threshold', type=float, default=0.7,
                        help='the amended max socre used to filterate boxes,default=0.7')
    args = parser.parse_args()

    amend = AmendImages(model_path=args.model_path, min_threshold=args.min_threshold, max_threshold=args.max_threshold)
    amend.amend_images()