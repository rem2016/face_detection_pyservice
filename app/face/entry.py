import cv2
import os  
os.environ['GLOG_minloglevel'] = '3'
import numpy as np
import time
from .insightface.deploy import face_model
import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mtcnn'))
from .mtcnn.Detection.MtcnnDetector import MtcnnDetector
from .mtcnn.Detection.detector import Detector
from .mtcnn.Detection.fcn_detector import FcnDetector
from .mtcnn.train_models.mtcnn_model import P_Net, R_Net, O_Net
from .mtcnn.prepare_data.loader import TestLoader

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='face/insightface/deploy/model-y1-test2/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
MODEL_WIDTH, MODEL_HEIGHT = 112, 112

class FaceEntry:
    def __init__(self, model='final', gpu=True, min_face_size = 24):
        # for detector
        thresh = [0.9, 0.6, 0.7]
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        prefix = ['mtcnn/data/MTCNN_model/PNet_landmark/PNet', 'mtcnn/data/MTCNN_model/RNet_landmark/RNet', 'mtcnn/data/MTCNN_model/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        batch_size = [2048, 256, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        model_path = [os.path.join(os.path.dirname(__file__), path) for path in model_path]
        # load pnet model
        if slide_window:
            PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
        else:
            print(model_path[0])
            PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet

        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

        self.mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

        self.model = face_model.FaceModel(args)

    def process_image(self, img_name):
        start_time = time.time()
        img = cv2.imread(img_name)
        img = self.model.get_input(img, self.mtcnn_detector)
        f1 = self.model.get_feature(img)
        end_time = time.time()
        total_cost = str(end_time - start_time)
        return f1, total_cost

    def process_image_from_bytes(self, bytes_img):
        start_time = time.time()
        img = cv2.imdecode(np.frombuffer(bytes_img, np.uint8), cv2.IMREAD_COLOR)
        img = self.model.get_input(img, self.mtcnn_detector)
        f1 = self.model.get_feature(img)
        end_time = time.time()
        total_cost = str(end_time - start_time)
        return f1, total_cost

    def extract_features_from_bytes(self, bytes_img, height=112, width=112):
        start_time = time.time()
        img = cv2.imdecode(np.frombuffer(bytes_img, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT))
        img = self.model.get_input_without_det(img, height, width)
        f1 = self.model.get_feature(img)
        end_time = time.time()
        total_cost = str(end_time - start_time)
        return f1, total_cost

    def _normalize_image(self, im):
        mean, std = cv2.meanStdDev(im)
        if std[0, 0] < 1e-6:
            std[0, 0] = 1
        nim = im * 1.0/std[0, 0] - 1*mean[0, 0]/std[0, 0]
        return nim


if __name__ == '__main__':
    face = FaceEntry()
    t0 = time.time()
    face.process_image('test.jpg')
    print(time.time() - t0)
    t0 = time.time()
    face.process_image('test.jpg')
    print(time.time() - t0)
