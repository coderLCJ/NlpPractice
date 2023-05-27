#! /usr/bin/env python
# -*- coding: utf-8 -*-
# author: laity
# date: 2023-05-26

from idl.Server import PredictServer
from idl.Server import ttypes

from handler import model_handler

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

__HOST = '127.0.0.1'
__PORT = 8080

class Handler(object):
    # 预填任务
    def __init__(self):
        self.model_handler = model_handler.Model()
        
    def predict(self, data):
        # 读取参数
        print('='*20, '\n接收参数')
        print(data)
        print('='*20, '\n读取参数')
        # 参数检测，取所需要的
        print('Have Models' if hasattr(data,"Models") else 'No Models')
        print('Have currentIds' if hasattr(data,"currentIds") else 'No currentIds') 
        inputs = data
        
        # 调用模型
        print('='*20, '\n调用模型') 
        errCode, errMsg, predictResults = self.model_handler.predict(inputs)
        
        # 返回结果
        print('='*20, '\n返回结果') 
        return ttypes.Response(errCode, errMsg, predictResults)


if __name__ == '__main__':
    handler = Handler()

    processor = PredictServer.Processor(handler)
    transport = TSocket.TServerSocket(__HOST, __PORT)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    rpcServer = TServer.TSimpleServer(processor,transport, tfactory, pfactory)

    print('Starting the rpc server at', __HOST,':', __PORT)
    rpcServer.serve()