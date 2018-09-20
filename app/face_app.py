# -*- coding: utf-8 -*-
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
from insightface.entry import FaceEntry
from common.face_common import FaceResource, FaceCommonJsonRet, FaceCodeMsg
import time
import numpy as np
import json

APP_URL_PREFIX = "/v1/api/face"
basepath = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在路径

app = Flask(__name__)

facer =  FaceEntry()

@app.route("/")
def hello():
    return "Hello World!"

#统一404处理
@app.errorhandler(404)
def page_not_not_found(error):
    return FaceCommonJsonRet(code=404,
                                success=False,
                                msg="404 Not Found . there is not this api",
                                data="").toJsonStr()


@app.route(APP_URL_PREFIX+"/health_check")
def health_check():
    return FaceCommonJsonRet(code=200,
                                success=True,
                                msg="health check is ok",
                                data="").toJsonStr()

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = basepath + '/static/uploads/' + secure_filename(f.filename)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(upload_path)
        f.save(upload_path)
        t0 = time.time()
        face_vector, cost_time = facer.process_image(upload_path)
        ret = FaceCommonJsonRet(code=FaceCodeMsg.SUCCESS.code,
                              success=True,
                              msg=FaceCodeMsg.SUCCESS.msg,
                              data=[face_vector, cost_time])
        return ret.toJson()

@app.route('/test', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        print(basepath)
        upload_path = basepath + '/static/uploads/Tom_Hanks_54745.png'
        print(upload_path)
        face_vector, cost_time = facer.process_image(upload_path)
        ret = FaceCommonJsonRet(code=FaceCodeMsg.SUCCESS.code,
                              success=True,
                              msg=FaceCodeMsg.SUCCESS.msg,
                              data=[str(face_vector), cost_time])
        return ret.toJsonStr()

@app.route('/test_bytes_trans', methods=['POST', 'GET'])
def test_bytes_trans():
    if request.method == 'POST':
        print(basepath)
        upload_path = basepath + '/static/uploads/Tom_Hanks_54745.png'
        print(upload_path)
        bytes_img_f = open(upload_path, "rb")
        print(bytes_img_f)
        bytes_img = np.fromfile(bytes_img_f, dtype=np.ubyte)
        print(bytes_img)
        face_vector, cost_time = facer.process_image_from_bytes(bytes_img)
        bytes_img_f.close()
        ret = FaceCommonJsonRet(code=FaceCodeMsg.SUCCESS.code,
                              success=True,
                              msg=FaceCodeMsg.SUCCESS.msg,
                              data=[str(face_vector), cost_time])
        return ret.toJsonStr()

@app.route('/detection_recognition', methods=['POST', 'GET'])
def detection_recognition():
    if request.method == 'POST':
        bytes_img = request.data
        face_vector, cost_time = facer.process_image_from_bytes(bytes_img)
        ret = FaceCommonJsonRet(code=FaceCodeMsg.SUCCESS.code,
                              success=True,
                              msg=FaceCodeMsg.SUCCESS.msg,
                              data=[str(face_vector), cost_time])
        return ret.toJsonStr()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=58481)
