import os  
os.environ['GLOG_minloglevel'] = '3'
import cv2
import caffe
import numpy as np

def normalize_image(im):
    mean, std = cv2.meanStdDev(im)
    if std[0, 0] < 1e-6:
            std[0, 0] = 1
    # nim = cv2.convertScaleAbs(im, cv2.CV_32F, 1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
    nim = im * 1.0/std[0, 0] - 1*mean[0, 0]/std[0, 0]
    print(1.0/std[0, 0], -1*mean[0, 0]/std[0, 0])
    return nim

def view_image(img_path, pts):
    im = cv2.imread(img_path)
    im = cv2.resize(im, (256, 256))
    for pt in pts:
        cv2.circle(im, ((int)(pt[0]), (int)(pt[1])), 1, (255, 255, 0), -1)
    cv2.imwrite(img_path[:-4] + '_res.jpg', im)

def pred_image(img_path, model='final'):
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    if (model == 'final'):
        net = caffe.Net('../../models/WFLW/WFLW_final/rel.prototxt', '../../models/WFLW/WFLW_final/model.bin', caffe.TEST)
    else:
        net = caffe.Net('../../models/WFLW/WFLW_wo_mp/rel.prototxt', '../../models/WFLW/WFLW_wo_mp/model.bin', caffe.TEST)
    gim = cv2.imread(img_path, 0)
    gim = cv2.resize(gim, (256, 256))
    # gim = cv2.convertScaleAbs(gim, cv2.CV_32F)
    gim = np.float32(gim)
    cv2.imwrite(img_path[:-4] + '_1nim.jpg', gim)
    nim = normalize_image(gim)
    cv2.imwrite(img_path[:-4] + '_nim.jpg', nim)
    net.blobs['data'].data[...] = nim
    out = net.forward()['result'][0].reshape((-1, 2))
    print(out)
    view_image(img_path, out)

pred_image('../../imgs/t.jpg')