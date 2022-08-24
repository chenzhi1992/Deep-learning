# -*- coding:utf-8 -*-
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os
import xml.etree.ElementTree as ET
import json
import random

import numpy as np
import argparse
import cv2
import sys
import PIL.Image
from shutil import copyfile
import redis

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
CLASS_NAMES = ["background", "Person", "Car", "Bicycle", "ElectroMboile", "MotorBike", "Tricycle", "Bus", "Truck", "Head","Rider"]
MyDataPath = '/chenzhi_cxzyzx/dataset/fcos_nonveh_modify'

# --------------- param ---------------------
parser = argparse.ArgumentParser(description='PyTorch CSRNet')
r = redis.Redis(host='10.68.4.10', port=6379, password='root', db=0)

parser.add_argument('--train_data_dir', default=None, help="the path of train_data_dir", nargs="?")
parser.add_argument('--preprocess_output_dir', default=None, help="the path of preprocess_output_dir", nargs="?")
parser.add_argument('--train_label_dir', default='', help="the path of train_label_dir", nargs="?")
parser.add_argument('--redis_key', default='', help='redis key', nargs="?")

args = parser.parse_args()
r.set(args.redis_key, 1)

bg_imgs = []
xj_bg_imgs = []
cut_obs = []
cut_ids = [3, 6, 8]
#cut_ids = [6]

#the path you want to save your results for coco to voc
img_dir = os.path.join(MyDataPath, 'coco_images')
anno_dir = os.path.join(MyDataPath, 'coco_annotations')

ch_img_dir = os.path.join(MyDataPath, 'chinese_images')
ch_anno_dir = os.path.join(MyDataPath, 'chinese_annotations')

# datasets_list=['train2014', 'val2014']
datasets_list=['train2017']

coco_filenames = []
select_class = ['person']
select_class2 = ['bicycle']
classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
#Store annotations and train2014/val2014/... in this folder
dataDir= '/coco/2017/'
 
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''

#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
mkr(img_dir)
mkr(anno_dir)
mkr(ch_img_dir)
mkr(ch_anno_dir)

def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(coco,dataset,filename,objs):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+ '/' + filename[:-3]+'xml'
    img_path=dataDir+dataset+'/'+filename
    # print(img_path)
    dst_imgpath=img_dir+ '/' + filename
 
    img=cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)
 
    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)
 
 
def showimg_dog(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    has_class = False
    I=PIL.Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            has_class = True
            # print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
    #             draw = ImageDraw.Draw(I)
    #             draw.rectangle([xmin, ymin, xmax, ymax])
    # if show:
    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(I)
    #     plt.show()
 
    return objs, has_class

def parse_coco_dog():
    for dataset in datasets_list:
        #./COCO/annotations/instances_train2014.json
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
    
        #COCO API for initializing annotated data
        coco = COCO(annFile)
        '''
        COCO 对象创建完毕后会输出如下信息:
        loading annotations into memory...
        Done (t=0.81s)
        creating index...
        index created!
        至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
        '''
        #show all classes in coco
        classes = id2name(coco)
        # print(classes)
        #[1, 2, 3, 4, 6, 8]
        classes_ids = coco.getCatIds(catNms=classes_names)
        # print(classes_ids)
        for cls in ['dog']:
            #Get ID number of this class
            cls_id=coco.getCatIds(catNms=[cls])
            img_ids=coco.getImgIds(catIds=cls_id)
            # print(cls,len(img_ids))
            # imgIds=img_ids[0:30000]
            for imgId in tqdm(img_ids):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                objs, has_class=showimg_dog(coco, dataset, img, classes,classes_ids,show=False)
                # print(objs)
                if has_class and filename not in coco_filenames:
                	save_annotations_and_imgs(coco, dataset, filename, objs)
                    
def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=PIL.Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
    #             draw = ImageDraw.Draw(I)
    #             draw.rectangle([xmin, ymin, xmax, ymax])
    # if show:
    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(I)
    #     plt.show()
 
    return objs

def parse_coco():
    for dataset in datasets_list:
        #./COCO/annotations/instances_train2014.json
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
    
        #COCO API for initializing annotated data
        coco = COCO(annFile)
        '''
        COCO 对象创建完毕后会输出如下信息:
        loading annotations into memory...
        Done (t=0.81s)
        creating index...
        index created!
        至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
        '''
        #show all classes in coco
        classes = id2name(coco)
        # print(classes)
        #[1, 2, 3, 4, 6, 8]
        classes_ids = coco.getCatIds(catNms=classes_names)
        # print(classes_ids)
        for cls in select_class:
            #Get ID number of this class
            cls_id=coco.getCatIds(catNms=[cls])
            img_ids=coco.getImgIds(catIds=cls_id)
            # print(cls,len(img_ids))
            imgIds=img_ids[0:20000]
            for imgId in tqdm(imgIds):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
                # print(objs)
                save_annotations_and_imgs(coco, dataset, filename, objs)
                coco_filenames.append(filename)
        for cls in select_class2:
            #Get ID number of this class
            cls_id=coco.getCatIds(catNms=[cls])
            img_ids=coco.getImgIds(catIds=cls_id)
            # print(cls,len(img_ids))
            imgIds=img_ids[0:5000]
            for imgId in tqdm(imgIds):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
                # print(objs)
                save_annotations_and_imgs(coco, dataset, filename, objs)
                coco_filenames.append(filename)

def cutmix(imga, imgb):
    #imga : 背景图片； imgb：提取的目标图片
    h_a, w_a, c_a = imga.shape
    h_b, w_b, c_b = imgb.shape
    # print(imga.shape)
    # print(imgb.shape)
    new_box = [] ##cut图片在背景图中的位置
    # if b.size > a.size, reduce b.size
    scale = 0
    if h_b >= h_a or w_b >= w_a:
        scale = max(float(h_b) / float(h_a), float(w_b) / float(w_a))
    scale = int(scale) + 1
    # print(w_b//scale, h_b//scale)
    imgb = cv2.resize(imgb, (w_b//scale, h_b//scale))
    h_b, w_b, c_b = imgb.shape
    x = random.randint(0, w_a - w_b)
    y = random.randint(0, h_a - h_b)
    imgc = imga * 0.5
    imgc[y:y+h_b, x:x+w_b,:] = imgc[y:y+h_b, x:x+w_b,:] + imgb * 0.5
    new_box.append(x)
    new_box.append(y)
    new_box.append(w_b)
    new_box.append(h_b)

    return imgc, new_box



#对比度和亮度
def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def add_gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

## 数据增强：对比度亮度调整，高斯模糊，高斯噪声,cutmix
def data_aug(img_path):
    try:
    	img = cv2.imread(img_path)
    except UnicodeEncodeError:
        return None
    aug_ind = random.randint(0, 1) ##随机生成一个数（前闭后闭），每个数代表一个数据增强方式
    isBlur =  random.randint(0, 1) ##是否进行高斯模糊，0 = 进行 ，1 = 不进行
    if aug_ind == 0: ## 降低对比度和亮度
        _alpha = random.uniform(0.5,1)
        _beta = random.uniform(-50,0)
        c_img = Contrast_and_Brightness(_alpha , _beta, img)
        if isBlur == 0:
            img_aug = cv2.GaussianBlur(c_img, (7,7), 2)
        else:
            img_aug = c_img
    elif aug_ind == 1:## ## 调高对比度和亮度
        _alpha = random.uniform(1, 1.5)
        _beta = random.uniform(0, 50)
        c_img = Contrast_and_Brightness(_alpha , _beta, img)
        if isBlur == 0:
            img_aug = cv2.GaussianBlur(c_img, (7,7), 2)
        else:
            img_aug = c_img
    # else: ## 添加高斯噪声
    #     g_nosie = add_gasuss_noise(img, 0, 0.001)
    #     if isBlur == 0:
    #         img_aug = cv2.GaussianBlur(g_nosie, (7,7), 2)
    #     else:
    #         img_aug = g_nosie
    return img_aug


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path, image_path, image_file_name, is_xj): 
    # parse image
    image = PIL.Image.open(image_path)

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
    current_image_id = addImgItem(file_name, size)

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
                if objName == u"行人" or objName == u"人":
                    objID = 1
                elif objName == u"骑手":
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
                if bbox[3] > 32 and objID != 9:
                    if objID in cut_ids:
                        cut_ob = [image_path, bbox, objID]
                        cut_obs.append(cut_ob)
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
                if objID == 9:
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
                #addAnnoItem(object_name, current_image_id, current_category_id, bbox )

            # if current_sub == 'sub_number':
            #     num_sub = int(current_sub.text)
            
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
                if bbox[3] > 32 and objID != 9:
                    if objID in cut_ids:
                        cut_ob = [image_path, bbox, objID]
                        cut_obs.append(cut_ob)
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
                if objID == 9:
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
                #addAnnoItem(object_name, current_image_id, current_category_id, bbox )

    #目标个数小于定值则作为背景图片
    if num_objs < 10:
        bg_img = [xml_path, image_path]
        bg_imgs.append(bg_img)
    if is_xj == 1 and num_objs < 20:
        bg_img = [xml_path, image_path]
        xj_bg_imgs.append(bg_img)



def parseXmlFiles_cut(xml_path, image_path, cut_obj, image_file_name): 
    # parse image
    image = PIL.Image.open(image_path)

    # image = PIL.Image.open(image_path)
    if image.format != 'JPEG':
        # print(image_file_name)
        # f_bad.write(image_file_name+ '\n')
        raise ValueError('Image format not JPEG')

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
    current_image_id = addImgItem(file_name, size)

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
                if objName == u"行人" or objName == u"人":
                    objID = 1
                elif objName == u"骑手":
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
                if bbox[3] > 32:
                    if objID in cut_ids:
                        cut_ob = [image_path, bbox, objID]
                        cut_obs.append(cut_ob)
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )

            # if current_sub == 'sub_number':
            #     num_sub = int(current_sub.text)
            
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
                    # if objID in cut_ids:
                    #     cut_ob = [image_path, bbox, objID]
                    #     cut_obs.append(cut_ob)
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
                #addAnnoItem(object_name, current_image_id, current_category_id, bbox )
    ## add cut img
    _object_name = CLASS_NAMES[cut_obj[2]]
    _current_category_id = cut_obj[2]
    _bbox = cut_obj[1]
    addAnnoItem(_object_name, current_image_id, _current_category_id, _bbox )


## 处理含有中文的图片和xml
def copy_chinese():
    num = 0
    chinese_num = 0
    
    image_root_path = '/XinShizongAlreadyTargetDetectionCheck/Image' + '/zhengchang/'
    xml_root_ptah = '/XinShizongAlreadyTargetDetectionCheck/Label' + '/zhengchang/'
    imgs = os.listdir(image_root_path)
    for img_path in imgs:
        print(num)
        num+=1
        img_full_path = os.path.join(image_root_path, img_path)
        xml_file_name = os.path.splitext(img_path)[0] + '.xml'
        xml_full_path = os.path.join(xml_root_ptah, xml_file_name)
        try:
            _img = cv2.imread(img_full_path)
        except UnicodeEncodeError:
            copyfile(img_full_path, ch_img_dir + '/ch_' + str(chinese_num) + '.jpg')
            copyfile(xml_full_path, ch_anno_dir + '/ch_' + str(chinese_num) + '.xml')
            chinese_num += 1
            print('chinese_num', chinese_num, img_path)

    return chinese_num


if __name__ == '__main__':
    
    #chin_num = copy_chinese()
    #print(chin_num)

    parse_coco()
    parse_coco_dog()

    anno_path = os.path.join(MyDataPath, 'annotations')
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    images_path = os.path.join(MyDataPath, 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    json_file_train = MyDataPath + '/annotations/instances_9_train.json'
    # json_file_test = args.preprocess_output_dir + '/annotations/instances_8_test.json'
    for cat in CLASS_NAMES:
        addCatItem(cat)
    print('==========category ==============')
    for k, v in category_set.items():
        print('category : (key : vale)', k, v)
    print('==========category ==============')


    ch_num = 0
    idx = 0
    # all_images = []
    data_aug_num = 0 ##用来设定多少次进行一次data_aug操作

    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image' + '/youfeijidongche/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label' + '/youfeijidongche/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
            
    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image' + '/noNonVeh/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label' + '/noNonVeh/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
    

    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehicleDataCollection/Image/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehicleDataCollection/Label/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
    
    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image/' + '/ID1318932/detect/shinei/ditie/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label/' + '/ID1318932/detect/shinei/ditie/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
    
    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image/' + '/ID1318932/detect/shinei/gaotie/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label/' + '/ID1318932/detect/shinei/gaotie/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
    
    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image/' + '/ID1318932/detect/shiwai/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label/' + '/ID1318932/detect/shiwai/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)

    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image/' + '/ID1318932/jinjing/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label/' + '/ID1318932/jinjing/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
    
    image_root_path = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Image/' + '/ID1318932/xj_changjing3/'
    xml_root_ptah = '/XinShiZong_TargetDetectionNon-motorVehiclesModification/Label/' + '/ID1318932/xj_changjing3/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.Jpeg'
        if not os.path.exists(image_root_path + image_file_name):
            image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
            if not os.path.exists(image_root_path + image_file_name):
                image_file_name = os.path.splitext(one_xml)[0] + '.jpeg'
        aug_image_name = os.path.splitext(one_xml)[0] + '_aug.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
        ## data augmentation
        data_aug_num += 1
        aug_img = data_aug(image_full_path)
        aug_img_path = MyDataPath + '/images/' + aug_image_name
        if data_aug_num % 3 == 0: ##1/3的概率
            try:
                cv2.imwrite(aug_img_path, aug_img)
            except UnicodeEncodeError:
                ch_num += 1
                continue
            parseXmlFiles(xml_full_path, image_full_path, aug_image_name, 0)
            idx += 1
            print(idx)
   
   
   
    
    ## mosaic
    image_root_path = '/chenzhi_cxzyzx/script/data-aug/dataset2/Image/'
    xml_root_ptah = '/chenzhi_cxzyzx/script/data-aug/dataset2/Label/'
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        # print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)

            
    ## cutmix ......
    print('cutmix start ...', len(cut_obs))
    cut_num = 0
    for i in range(len(cut_obs)):
        cut_ob = cut_obs[i]
        ## 小的目标不做cutmix
        if cut_ob[1][2] < 50 or cut_ob[1][3] < 50:
            continue
        try:
            _img = cv2.imread(cut_ob[0])
        except UnicodeEncodeError:
            continue
        cut_img = _img[cut_ob[1][1]:cut_ob[1][1]+cut_ob[1][3], cut_ob[1][0]:cut_ob[1][0]+cut_ob[1][2]]
        bg_ind = random.randint(0, len(bg_imgs)-1)
        # print(bg_imgs[bg_ind][1])
        try:
            bg_img = cv2.imread(bg_imgs[bg_ind][1])
        except UnicodeEncodeError:
            continue
        new_img, new_box = cutmix(bg_img, cut_img)
        # cut_ob[1] = new_box
        new_ob = []
        new_ob.append(cut_ob[0])
        new_ob.append(new_box)
        new_ob.append(cut_ob[2])

        new_path = 'cutmix_' + str(cut_num) + '.jpg'
        new_img_path = MyDataPath + '/images/' + new_path
        cv2.imwrite(new_img_path, new_img)
        parseXmlFiles_cut(bg_imgs[bg_ind][0], bg_imgs[bg_ind][1], new_ob, new_path)
        cut_num += 1
    print('cut number..', cut_num)
 

    ## cutmix xj ......
    print('xj cutmix num = ', len(xj_bg_imgs))
    for i in range(len(xj_bg_imgs)):
        obj_ran = random.randint(0, len(cut_obs) -1)
        cut_ob = cut_obs[obj_ran]
        ## 小的目标不做cutmix
        # if cut_ob[1][2] < 50 or cut_ob[1][3] < 50:
        #     continue
        try:
            _img = cv2.imread(cut_ob[0])
        except UnicodeEncodeError:
            continue
        cut_img = _img[cut_ob[1][1]:cut_ob[1][1]+cut_ob[1][3], cut_ob[1][0]:cut_ob[1][0]+cut_ob[1][2]]
        bg_ind = random.randint(0, len(xj_bg_imgs)-1)
        # print(bg_imgs[bg_ind][1])
        try:
            bg_img = cv2.imread(xj_bg_imgs[bg_ind][1])
        except UnicodeEncodeError:
            continue
        new_img, new_box = cutmix(bg_img, cut_img)
        # cut_ob[1] = new_box
        new_ob = []
        new_ob.append(cut_ob[0])
        new_ob.append(new_box)
        new_ob.append(cut_ob[2])

        new_path = 'cutmix_' + str(cut_num) + '.jpg'
        new_img_path = MyDataPath + '/images/' + new_path
        cv2.imwrite(new_img_path, new_img)
        parseXmlFiles_cut(xj_bg_imgs[bg_ind][0], xj_bg_imgs[bg_ind][1], new_ob, new_path)
        cut_num += 1
    print('cut number 3..', cut_num)

    ## coco data....
    image_root_path = img_dir
    xml_root_ptah = anno_dir
    xmls = os.listdir(xml_root_ptah)

    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        #print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)
    
    ## chinese path data....
    image_root_path = ch_img_dir
    xml_root_ptah = ch_anno_dir
    xmls = os.listdir(xml_root_ptah)
    for one_xml in xmls:
        xml_full_path = os.path.join(xml_root_ptah, one_xml)
        # print(xml_full_path)
        image_file_name = os.path.splitext(one_xml)[0] + '.jpg'
        image_full_path = os.path.join(image_root_path, image_file_name)
        #print(image_full_path)

        copyfile(image_full_path, MyDataPath + '/images/' + image_file_name)
        parseXmlFiles(xml_full_path, image_full_path, image_file_name, 0)
        idx += 1
        print(idx)


    # print(coco)
    json.dump(coco, open(json_file_train, 'w'))
    # json.dump(coco_test, open(json_file_test, 'w'))
    print('ch_num = ', ch_num)
    #print('ch_path_num=', chin_num)
    print('-->create json is done')

    r.set(args.redis_key, 0)
