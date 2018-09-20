from deploy import face_model
import argparse
import cv2
import sys
import numpy as np
import time

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

class FaceEntry:
    def __init__(self):
        self.model = face_model.FaceModel(args)

    def process_image(self, img_name):
        start_time = time.time()
        img = cv2.imread(img_name)
        img = self.model.get_input(img)
        f1 = self.model.get_feature(img)
        end_time = time.time()
        total_cost = str(end_time - start_time)
        return f1, total_cost

    def process_image_from_bytes(self, bytes_img):
        start_time = time.time()
        print(1)
        img = cv2.imdecode(np.fromstring(bytes_img, np.uint8), 1)
        print(1)
        img = self.model.get_input(img)
        print(1)
        f1 = self.model.get_feature(img)
        print(1)
        end_time = time.time()
        print(1)
        total_cost = str(end_time - start_time)
        print(1)
        return f1, total_cost

    def _normalize_image(self, im):
        mean, std = cv2.meanStdDev(im)
        if std[0, 0] < 1e-6:
            std[0, 0] = 1
        # nim = cv2.convertScaleAbs(im, cv2.CV_32F, 1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
        nim = im * 1.0/std[0, 0] - 1*mean[0, 0]/std[0, 0]
        # print(1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
        return nim
facer = FaceEntry()

if __name__ == '__main__':
    face = FaceEntry()
    t0 = time.time()
    face.process_image('test.jpg')
    print(time.time() - t0)
    t0 = time.time()
    face.process_image('test.jpg')
    print(time.time() - t0)
