#encoding=utf8
import json
from flask_restplus import Resource
from enum import Enum

class FaceEnvEnum(Enum):
    #开发环境
    DEVELOP = "develop"
    #生产环境
    PRODUCTION = "production"


#服务统一返回接口格式
class FaceCommonJsonRet():
    def __init__(self,code ,success, msg ,data ):
        self.code = code
        self.msg = msg
        self.data = data
        self.success = success

    def toJsonStr(self):
        return json.dumps(self.__dict__)

    def toJson(self):
        return self.__dict__


class FaceCodeMsg():
    class CM():
        def __init__(self,code,msg):
            self.code = code
            self.msg = msg

    SUCCESS = CM(200,"success")
    RECOG_ERROR = CM(300, "recognition failed")

class FaceResource(Resource):
    def __init__(self , api=None, *args, **kwargs):
        # todo
        super(FaceResource, self).__init__(api, args, kwargs)


