# -*- coding:utf-8 -*-
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os
import xml.etree.ElementTree as ET
import json
import random
from random import shuffle

import numpy as np
import argparse
import cv2
import sys

from shutil import copyfile
import redis

from PIL import Image,ImageDraw
import math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 20180000000
annotation_id = 0
CLASS_NAMES = ["background", "Person", "Car", "Bicycle", "ElectroMboile", "MotorBike", "Tricycle", "Bus", "Truck", "Head"]

# --------------- param ---------------------
parser = argparse.ArgumentParser(description='PyTorch CSRNet')
r = redis.Redis(host='10.68.4.10', port=6379, password='root', db=0)

parser.add_argument('--train_data_dir', default=None, help="the path of train_data_dir", nargs="?")
parser.add_argument('--preprocess_output_dir', default=None, help="the path of preprocess_output_dir", nargs="?")
parser.add_argument('--train_label_dir', default='', help="the path of train_label_dir", nargs="?")
parser.add_argument('--redis_key', default='', help='redis key', nargs="?")

args = parser.parse_args()
r.set(args.redis_key, 1)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2
    image_datas = []
    box_datas = []
    index = 0
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)-0.1
        print('scale=', scale)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


def normal_(annotation_line, input_shape):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    iw, ih = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    box[:, [0, 2]] = iw - box[:, [2, 0]]

    return image, box

def parseXmlFiles(xml_path, image_path, image_file_name): 
    box_txt = ''
    # parse image
    image = Image.open(image_path)

    # image = PIL.Image.open(image_path)
    if image.format != 'JPEG':
        # raise ValueError('Image format not JPEG')
        print('Image format not JPEG')

    num_objs = 0 ##用于记录每个图片中的目标个数，小于定值则作为背景图片

    bndbox = dict()
    size = dict()
    current_image_id = None
    current_category_id = None
    file_name = None
    size['width'] = None
    size['height'] = None
    size['depth'] = None

    width, height = image.size
    size['width'] = width
    size['height'] = height
    size['depth'] = 3
    file_name = image_file_name

    # parse xml
    tree = ET.parse(xml_path)
    root =  tree.getroot()
    if root.tag != 'annotation':
        raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
    
     #elem is <folder>, <filename>, <size>, <object>
    for elem in root:
        current_parent = elem.tag
        current_sub = None
        objName = None
        
        if elem.tag != 'object':
            continue
        objID = -1
        for subelem in elem:
            bndbox ['xmin'] = None
            bndbox ['xmax'] = None
            bndbox ['ymin'] = None
            bndbox ['ymax'] = None
            
            current_sub = subelem.tag
            if current_parent == 'object' and subelem.tag == 'name':
                objName = subelem.text
                if objName == u"行人" or objName == u"人" or objName == u"骑手":
                    objID = 1
                elif objName == u"轿车" or objName == u"汽车" or objName == u"小轿车":
                    objID = 2
                elif objName == u"SUV":
                    objID = 2
                elif objName == u"MPV":
                    objID = 2
                elif objName == u"跑车":
                    objID = 2
                elif objName == u"皮卡":
                    objID = 2
                elif objName == u"面包车":
                    objID = 2
                elif objName == u"自行车":
                    objID = 3
                elif objName == u"电瓶车":
                    objID = 4
                elif objName == u"摩托车":
                    objID = 5
                elif objName == u"三轮车":
                    objID = 6
                elif objName == u"小客":
                    objID = 7
                elif objName == u"中客":
                    objID = 7
                elif objName == u"大客" or objName == u"客车":
                    objID = 7
                elif objName == u"微型卡车":
                    objID = 8
                elif objName == u"轻型卡车":
                    objID = 8
                elif objName == u"中型卡车":
                    objID = 8
                elif objName == u"重型卡车" or objName == u"卡车":
                    objID = 8
                elif objName == u"头肩" or objName == u"头肩人脸" or objName == u"头肩脸":
                    objID = 9
                elif objName == 'mask':
                    objID = 0
                elif objName == 'person' or objName == 'Person':
                    objID = 1
                elif objName == 'car' or objName == 'Car':
                    objID = 2
                elif objName == 'bicycle' or objName == 'Bicycle':
                    objID = 3
                elif objName == 'ElectroMboile':
                    objID = 4
                elif objName == 'motorcycle' or objName == 'MotorBike':
                    objID = 5
                elif objName == 'Tricycle':
                    objID = 6
                elif objName == 'bus' or objName == 'Bus':
                    objID = 7
                elif objName == 'truck' or objName == 'Truck':
                    objID = 8
                elif objName == 'Head':
                    objID = 9
                else:
                    print(image_path)
                num_objs += 1
            if objID == 0:
                objID = -1
                break

            if current_sub == 'bndbox':
                for option in subelem:
                    bndbox[option.tag] = int(option.text)
                bbox = []
                bbox.append(bndbox ['xmin'])
                bbox.append(bndbox ['ymin'])
                bbox.append(bndbox['xmax'] - bndbox['xmin'])
                bbox.append(bndbox['ymax'] - bndbox['ymin'])
                object_name = CLASS_NAMES[objID]
                current_category_id = objID

                ##剔除掉小的目标 ,规定面积在1024(32*32)以下为小目标
                #if bbox[2] * bbox[3] >= 1024:
                #if bbox[3] > 32:
                if bbox[3] > 32:
                    box_txt += (str(bndbox ['xmin']) + ',' + str(bndbox ['ymin']) + ',' + str(bndbox ['xmax']) + ',' + str(bndbox ['ymax']) + ',' + str(objID) + ' ')

            
            if 'subbox' in current_sub:
                for option in subelem:
                    bndbox[option.tag] = int(option.text)
                bbox = []
                bbox.append(bndbox ['xmin'])
                bbox.append(bndbox ['ymin'])
                bbox.append(bndbox['xmax'] - bndbox['xmin'])
                bbox.append(bndbox['ymax'] - bndbox['ymin'])
                object_name = CLASS_NAMES[objID]
                current_category_id = objID

                ##剔除掉小的目标 ,规定面积在1024(32*32)以下为小目标
                #if bbox[2] * bbox[3] >= 1024:
                if bbox[3] > 32:
                    box_txt += (str(bndbox ['xmin']) + ',' + str(bndbox ['ymin']) + ',' + str(bndbox ['xmax']) + ',' + str(bndbox ['ymax']) + ',' + str(objID) + ' ')

    # #目标个数小于定值则作为背景图片
    # if num_objs < 10:
    #     bg_img = [xml_path, image_path]
    #     bg_imgs.append(bg_img)
    return box_txt


if __name__ == '__main__':
    

    image_root_path = '/MultipleSceneNineDetection/Image' + '/shequchurukou/'
    xml_root_ptah = '/MultipleSceneNineDetection/Label' + '/shequchurukou/'
    xmls = os.listdir(xml_root_ptah)
    shuffle(xmls)
    print('xml num1 = ', len(xmls))
    annos = []
    ind = 0
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        boxes_txt = parseXmlFiles(xml_full_path, image_full_path, image_file_name)
        anno_txt = (image_full_path + ' ' + boxes_txt[:-1])
        print(anno_txt)
        annos.append(anno_txt)
        if (len(annos) % 4 == 0):
            image_data, box_data = get_random_data(annos, [1080, 1920])
            img = Image.fromarray((image_data * 255).astype(np.uint8))
            for j in range(len(box_data)):
                thickness = 3
                left, top, right, bottom = box_data[j][0:4]
                draw = ImageDraw.Draw(img)
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
            #img.show()
            save_path = './' + str(ind) + "_res.jpg"
            img.save(save_path)
            ind += 1
            annos.clear()
            break
