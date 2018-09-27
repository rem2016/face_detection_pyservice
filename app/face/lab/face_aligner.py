import os  
os.environ['GLOG_minloglevel'] = '3'
import cv2
import caffe
import numpy as np
import time


class FaceAligner:
    def __init__(self, model='final', gpu=True):
        if not gpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        self.model = model
        if (model == 'final'):
            self.net = caffe.Net('models/WFLW/WFLW_final/rel.prototxt', 'models/WFLW/WFLW_final/model.bin', caffe.TEST)
        else:
            self.net = caffe.Net('models/WFLW/WFLW_wo_mp/rel.prototxt', 'models/WFLW/WFLW_wo_mp/model.bin', caffe.TEST)

    def pred_image(self, img_path):
        gim = cv2.imread(img_path, 0)
        gim = cv2.resize(gim, (256, 256))
        gim = np.float32(gim)
        # cv2.imwrite(img_path[:-4] + '_1nim.jpg', gim)
        nim = self._normalize_image(gim)
        # cv2.imwrite(img_path[:-4] + '_nim.jpg', nim)
        self.net.blobs['data'].data[...] = nim
        out = self.net.forward()['result'][0].reshape((-1, 2))
        print(out)
        self.view_image(img_path, out)
        return out

    def _normalize_image(self, im):
        mean, std = cv2.meanStdDev(im)
        if std[0, 0] < 1e-6:
            std[0, 0] = 1
        # nim = cv2.convertScaleAbs(im, cv2.CV_32F, 1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
        nim = im * 1.0/std[0, 0] - 1*mean[0, 0]/std[0, 0]
        # print(1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
        return nim

    def view_image(self, img_path, pts):
        im = cv2.imread(img_path)
        im = cv2.resize(im, (256, 256))
        for pt in pts:
            cv2.circle(im, ((int)(pt[0]), (int)(pt[1])), 1, (255, 255, 0), -1)
        cv2.imwrite(img_path[:-4] + '_' + self.model + '.jpg', im)

if __name__ == '__main__' :
    # img_path = 'imgs/t.jpg'
    aligner = FaceAligner(model='final', gpu=True)
    # aligner.pred_image(img_path)
    t = 0
    for i in range(1, 13):
        img_path = 'imgs/acne/' + str(i) + '.jpg'
        t1 = time.time()
        aligner.pred_image(img_path)
        t2 = time.time()
        t += t2 - t1
    print(t * 1.0 / 12)
