# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import os
import cv2
import numpy as np
import time
from common import face_preprocess
import sklearn
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fmobilefacenet
from mtcnn_detector import MtcnnDetector
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

def cal_time_cost(embedding, detector, loop_time):
    model = mx.mod.Module(symbol=embedding, context=mx.cpu(0), label_names=None)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.init_params()
    read_img_time = 0
    crop_time = 0
    embedding_time = 0

    for i in range(loop_time):
        # 读取图片
        start_time = time.time()
        img = cv2.imread('Tom_Hanks_54745.png')
        end_time = time.time()
        read_img_time += end_time - start_time
        #print('cost of image read:' + str(end_time - start_time))

    for i in range(loop_time):
        # 从图片中crop人脸
        start_time = time.time()
        ret = detector.detect_face(img, det_type = 0)
        bbox, points = ret
        bbox = bbox[0,0:4]
        points = points[0,:].reshape((2,5)).T
        nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        end_time = time.time()
        crop_time += end_time - start_time
        #print('cost of crop input:' + str(end_time - start_time))

    for i in range(loop_time):
        # 根据人脸获取embedding
        start_time = time.time()
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        #embedding = sklearn.preprocessing.normalize(embedding).flatten()
        end_time = time.time()
        embedding_time += end_time - start_time
        #print('cost of generate features:' + str(end_time - start_time))

    return read_img_time/loop_time, crop_time/loop_time, embedding_time/loop_time

ave_image_read_dict = {}
ave_crop_dict = {}
ave_embedding_dict = {}


# 原始模型
embedding = fmobilefacenet.get_symbol(128, bn_mom = 0.9, version_output = 'GNAP')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker=1,
                         accurate_landmark = True, threshold=[0.6,0.7,0.8])
ave_read_image_time, ave_crop_time, ave_embedding_time = cal_time_cost(embedding, detector, 50)
print(ave_read_image_time, ave_crop_time, ave_embedding_time)
ave_image_read_dict['orignal'] = ave_read_image_time
ave_crop_dict['orignal'] = ave_crop_time
ave_embedding_dict['orignal'] = ave_embedding_time



# 去掉45，5层
embedding = fmobilefacenet.get_symbol1(128, bn_mom = 0.9, version_output = 'GNAP')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker=1,
                         accurate_landmark = True, threshold=[0.6,0.7,0.8])
ave_read_image_time, ave_crop_time, ave_embedding_time = cal_time_cost(embedding, detector, 50)
print(ave_read_image_time, ave_crop_time, ave_embedding_time)
ave_image_read_dict['rm45,5'] = ave_read_image_time
ave_crop_dict['rm45,5'] = ave_crop_time
ave_embedding_dict['rm45,5'] = ave_embedding_time



# 去掉34，4层
embedding = fmobilefacenet.get_symbol2(128, bn_mom = 0.9, version_output = 'GNAP')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker=1,
                         accurate_landmark = True, threshold=[0.6,0.7,0.8])
ave_read_image_time, ave_crop_time, ave_embedding_time = cal_time_cost(embedding, detector, 50)
print(ave_read_image_time, ave_crop_time, ave_embedding_time)
ave_image_read_dict['rm34,4'] = ave_read_image_time
ave_crop_dict['rm34,4'] = ave_crop_time
ave_embedding_dict['rm34,4'] = ave_embedding_time



# 去掉23，3层
embedding = fmobilefacenet.get_symbol3(128, bn_mom = 0.9, version_output = 'GNAP')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(0), num_worker=1,
                         accurate_landmark = True, threshold=[0.6,0.7,0.8])
ave_read_image_time, ave_crop_time, ave_embedding_time = cal_time_cost(embedding, detector, 50)
print(ave_read_image_time, ave_crop_time, ave_embedding_time)
ave_image_read_dict['rm23,3'] = ave_read_image_time
ave_crop_dict['rm23,3'] = ave_crop_time
ave_embedding_dict['rm23,3'] = ave_embedding_time

for k in ave_image_read_dict:
    print(k + ":" + ave_image_read_dict[k] + ", " + str(ave_crop_dict[k]) + ", " +  str(ave_embedding_dict[k]))