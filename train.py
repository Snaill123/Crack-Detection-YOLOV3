import argparse
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from voc_annotation import trainsform_data
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


K.tensorflow_backend.set_session(get_session())


class MyTrain(object):
    """训练器"""
    annotation_path = 'model_data/2007_train.txt'
    log_dir = 'logs/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    input_shape = (416, 416)

    def __init__(self, usage='train'):
        """

        Args:
            usage: 'train' / 'amend', 用于指定训练 训练集 还是从 修正数据集
        """
        trainsform_data(usage)  # 进行数据集转换,从voc到coco

    @staticmethod
    def _my_callback():
        """回调函数"""
        logging = TensorBoard(log_dir=MyTrain.log_dir)
        checkpoint = ModelCheckpoint(MyTrain.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
        return [logging, checkpoint, reduce_lr]

    @staticmethod
    def _create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
                      weights_path='Weights/11.25/test.h5'):
        """create the training model"""
        K.clear_session()
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, num_classes + 5)) for l in range(3)]

        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers) - 3)[freeze_body - 1]
                for i in range(num):
                    model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
                            )([*model_body.output, *y_true])
        return Model([model_body.input, *y_true], model_loss)

    @staticmethod
    def _get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def _get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def _valid_split(valid_rate=0.1):
        """训练集划分"""
        with open(MyTrain.annotation_path) as f:
            lines = f.readlines()
        np.random.shuffle(lines)
        num_val = int(len(lines) * valid_rate)
        num_train = len(lines) - num_val
        return num_train, num_val, lines

    @staticmethod
    def _data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
        """data generator for fit_generator"""
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def _data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0:
            return None
        return self._data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def start_train(self, val_split=0.1, batch_size=4, load_pretrained=False, epoch_1=20, epoch_2=40, weight_path=None):
        """构建计算图开始训练

        Args:
            val_split(float): 验证集比例
            batch_size(int): batch_size
            load_pretrained(bool): 是否预加载训练模型,
            epoch_1: 冻结卷积层训练次数
            epoch_2: 不冻结卷积层训练次数
            weight_path: 如果选择加载预训练模型,需要制定模型路径

        Examples:
            ```
              train = MyTrain()

              train.start_train(load_pretrained=True, weight_path='model_data/weights_pretrained.h5')
            ```

        """

        class_names = self._get_classes(MyTrain.classes_path)
        num_classes = len(class_names)
        anchors = self._get_anchors(MyTrain.anchors_path)
        model = self._create_model(MyTrain.input_shape, anchors, num_classes, freeze_body=2,
                                   load_pretrained=load_pretrained, weights_path=weight_path)
        num_train, num_val, lines = self._valid_split(val_split)
        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if True:
            # use custom yolo_loss Lambda layer.
            model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            model.fit_generator(self._data_generator_wrapper(lines[:num_train], batch_size,
                                                             MyTrain.input_shape, anchors, num_classes),
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=self._data_generator_wrapper(lines[num_train:], batch_size,
                                                                          MyTrain.input_shape, anchors, num_classes),
                                validation_steps=max(1, num_val // batch_size),
                                epochs=epoch_1,
                                initial_epoch=0,
                                callbacks=self._my_callback())
            model.save_weights(MyTrain.log_dir + 'trained_weights_stage_1.h5')
        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('Unfreeze all of the layers.')
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(self._data_generator_wrapper(
                                lines[:num_train], batch_size, MyTrain.input_shape, anchors, num_classes),
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=self._data_generator_wrapper(lines[num_train:], batch_size,
                                                                             MyTrain.input_shape, anchors, num_classes),
                                validation_steps=max(1, num_val // batch_size),
                                epochs=epoch_2,
                                initial_epoch=epoch_1,
                                callbacks=self._my_callback())
            model.save_weights(MyTrain.log_dir + 'test.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usage', type=str, choices=['train', 'amend'], default='train',
                        help='training or amending the model, default=train')
    parser.add_argument('-v', '--valid_rate', type=float, default=0.1,
                        help='the proportion of validation set, default=0.1 ')
    parser.add_argument('--load_model', type=bool, default=True,
                        help='whether to load the pretrained model,'
                        'if load_model=True, you need to assign the weight path')
    parser.add_argument('--weight_path', type=str, default='model_data/weights_pretrained.h5',
                        help='the pretrained weights path, it only valid when the argument `--load_model=True`')
    parser.add_argument('--epoch_1', type=int, default=10, help="the  training epochs of  frozen layers model")
    parser.add_argument('--epoch_2', type=int, default=40, help="the  training epochs of  unfreeze layers model")
    parser.add_argument('--batch_size', type=int, default=4, help='the batchsize of training')
    args = parser.parse_args()

    train = MyTrain(usage=args.usage)
    train.start_train(load_pretrained=args.load_model,
                      val_split=args.valid_rate,
                      batch_size=args.batch_size,
                      epoch_1=args.epoch_1,
                      epoch_2=args.epoch_2,
                      weight_path=args.weight_path)
