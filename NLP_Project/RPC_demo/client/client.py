#! /usr/bin/env python
# -*- coding: utf-8 -*-
# author: laity
# date: 2023-05-26

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


from idl.Server.PredictServer import Client
from idl.Server import ttypes

__HOST = '127.0.0.1'
__PORT = 8080

tsocket = TSocket.TSocket(__HOST, __PORT)
transport = TTransport.TBufferedTransport(tsocket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Client(protocol)


if __name__ == '__main__':
    # 构造输入
    Models = ['BERT', 'T5']
    currentId = 0
    requestType = 1
    extendInfo = {'BERT': 'Hello BERT', 'T5': 'Hello T5'}
    data = ttypes.Request(Models, currentId, requestType, extendInfo)
    transport.open()

    print(data)
    print('\n[respose]\n')
    print(client.predict(data))