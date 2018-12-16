import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from os import getcwd


sets=[('2007', 'train')]
classes = ["neg"]


def _write_data():
    xmlfilepath = 'DATA/VOC2007/Annotations'
    txtsavepath = 'DATA/VOC2007/ImageSets/Main'
    all_images = [i.split('.')[0] for i in os.listdir(xmlfilepath)]
    with open(os.path.join(txtsavepath, 'train.txt'), 'w') as train_f:
        for i in all_images:
            train_f.write(str(i)+'\n')
def _write_amend_data():
    xmlfilepath = 'DATA/Amend_VOC2007/Annotations'
    txtsavepath = 'DATA/Amend_VOC2007/ImageSets/Main'
    all_images = [i.split('.')[0] for i in os.listdir(xmlfilepath)]
    with open(os.path.join(txtsavepath, 'train.txt'), 'w') as train_f:
        for i in all_images:
            train_f.write(str(i) + '\n')


def _convert_annotation(year, image_id, list_file):
    in_file = open('DATA/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
def _convert_amend_annotation(year, image_id, list_file):
    in_file = open('DATA/Amend_VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def trainsform_data(usage='train'):
    wd = getcwd()
    if usage == 'train':
        _write_data()
        for year, image_set in sets:
            image_ids = open('DATA/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
            list_file = open(os.path.join('model_data', '%s_%s.txt' % (year, image_set)), 'w')
            for image_id in tqdm(image_ids, desc="开始转换数据集[%s]" % usage):
                list_file.write('%s/DATA/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
                _convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
    elif usage == 'amend':
        _write_amend_data()
        for year, image_set in sets:
            image_ids = open('DATA/Amend_VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
            list_file = open(os.path.join('model_data', '%s_%s.txt' % (year, image_set)), 'w')
            for image_id in tqdm(image_ids, desc="开始转换数据集[%s]" % usage):
                list_file.write('%s/DATA/Amend_VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
                _convert_amend_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
    else:
        raise print("请输入正确数据集用途")


if __name__ == '__main__':
    trainsform_data()