# -*- coding: utf-8 -*-
from flask import Flask,render_template,request,redirect,url_for
from functools import wraps
import traceback
import warnings
from werkzeug.utils import secure_filename
from flask_restplus import Resource, Api, Namespace
import os
#from face.insightface.entry import FaceEntry
from face.entry import FaceEntry
from common.face_common import FaceResource, FaceCommonJsonRet, FaceCodeMsg
import time
import numpy as np
import json
from flask_restplus import reqparse

basepath = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在路径
app = Flask(__name__)
base_ns = Namespace("base_api", description="base_api doc and test")
api_plus = Api(app,version="v1.0.0",title='Face_detection_server')
api_plus.add_namespace(base_ns)
facer = FaceEntry()

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


@app.route("/health_check")
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


def api_exception500_wrap(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            warnings.warn(str(e), UserWarning)
            traceback.print_exc()
            ret = (dict(msg=str(e)), 500)
        return ret
    return new_func


@api_plus.route('/detect/imgsize/<string:imgsize>')
class Detect(FaceResource):
    @api_exception500_wrap
    def post(self, imgsize):
        bytes_img = request.data
        face_vector, cost_time = facer.process_image_from_bytes(bytes_img)
        return [dict(code=FaceCodeMsg.SUCCESS.code,
                        success=True,
                        msg=FaceCodeMsg.SUCCESS.msg,
                        feature=face_vector.tolist(),
                        cost_time=cost_time)]


@api_plus.route("/recognize/imgsize/<string:imgsize>")
class Recognize(FaceResource):
    @api_exception500_wrap
    def post(self, imgsize):
        width, height = imgsize.split('x')
        height = int(height)
        width = int(width)
        bytes = request.data

        face_vector, cost_time = facer.extract_features_from_bytes(bytes, height, width)
        return dict(code=FaceCodeMsg.SUCCESS.code,
                    success=True,
                    msg=FaceCodeMsg.SUCCESS.msg,
                    feature=face_vector.tolist(),
                    cost_time=cost_time)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
