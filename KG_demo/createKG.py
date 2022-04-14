# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         createKG
# Description:  构造知识图谱
# Author:       Laity
# Date:         2021/10/21
# ---------------------------------------------
from py2neo import Graph,Node,Relationship
import os

classification = ['花卉类别', '花卉功能', '应用环境', '盛花期_习性', '养护难度']  # 总类别
page_size = [0, 12, 20, 34, 42, 46]  # 页面范围

def createEntity0(graph):
    # step 0 总节点
    # ------------------------------------------------------
    cql = 'CREATE (:花卉大全{id:\'0\', name:\'花卉大全\'})'
    graph.run(cql)
    for i, c in enumerate(classification):
        cql = '''
            MERGE (a:花卉大全{id:\'%d\', name:\'%s\'})
            MERGE (b {name: '花卉大全'}) 
            MERGE (b)-[:划分]-> (a)
            ''' % (i+1, c)
        graph.run(cql)
    print('step 0 done')
    # ------------------------------------------------------
    # step 1 类细分
    # ------------------------------------------------------
    content_file = open('data/花卉大全.txt', 'r', encoding='utf8')
    for i in range(5):
        for j in range(page_size[i], page_size[i+1]):
            name = content_file.readline().split()[-1]
            cql = '''
                MERGE (a:%s{id:\'%d\', name:\'%s\'})
                MERGE (b {name: '%s'}) 
                MERGE (b)-[:划分]-> (a)
            ''' % (classification[i], j, name, classification[i])
            graph.run(cql)
    print('step 1 done')
    # ------------------------------------------------------
    # step 2 生成品种
    # ------------------------------------------------------
    cql = 'CREATE (:花卉品种{id:\'0\', name:\'花卉品种\'})'
    graph.run(cql)
    file = open('data/种类.txt', 'r', encoding='utf8')
    i = 1
    for name in file.readlines():
        if len(name) == 0:
            continue
        cql = '''
                MERGE (a:花卉品种{id:\'%d\', name:\'%s\'})
                MERGE (b {name: '花卉品种'})
                MERGE (b)-[:划分]-> (a)
            ''' % (i, name)
        i += 1
        graph.run(cql)
    print('step 2 done')
    # ------------------------------------------------------
    # step 3 分界
    # ------------------------------------------------------
    belong = ['界', '门', '纲', '目', '科', '属', '种']
    for i, name in enumerate(belong):
        cql = 'CREATE (:生物学分支{id:\'%d\', name:\'%s\'})' % (i, name)
        graph.run(cql)
        if i > 0:
            cql = '''
                MERGE (a {name: '%s'})
                MERGE (b {name: '%s'})
                MERGE (b)-[:划分]-> (a)
            ''' % (belong[i], belong[i-1])
            graph.run(cql)
    file_path = 'data/科属/'
    i = 0
    for p in belong:
        path = file_path + p + '.txt'
        for line in open(path, 'r', encoding='utf8').readlines():
            line = line.strip()
            if len(line) > 0 and p[0] == line[-1]:
                cql = '''
                    MERGE (a:具体分支{id:\'%d\', name:\'%s\'})
                    MERGE (b {name: '%s'})
                    MERGE (a)-[:属于]-> (b)
                ''' % (i, line, p)
                graph.run(cql)
    print('step 3 done')
    print('Entity构造完成')

def createFlower(graph):
    path2_file = open('data/花卉大全.txt', 'r', encoding='utf8')
    delt = ['\'', ')', '(', '{', '}']

    for i, path1 in enumerate(classification):
        for j in range(page_size[i], page_size[i+1]):
            class_name = path2_file.readline().split()[-1]
            flower_path = 'data/' + path1 + '/' + class_name + '.txt'
            flower = open(flower_path, 'r', encoding='utf8')
            # 读取详细内容
            for line in flower.readlines():
                if len(line.split('\t')) != 8:
                    print('error')
                    continue
                id, name, another_name, img, class_, belong, open_time, desc = line.split('\t')
                belongs = belong.split()
                belong_to = belongs[-1]
                for d in delt:
                    desc = desc.replace(d, ' ')
                    name = name.replace(d, ' ')
                    another_name = another_name.replace(d, ' ')


                cql = '''
                    MERGE (a:花卉{id:\'%d\', name:\'%s\', 别名:\'%s\', 图片:\'%s\', 开花季节:\'%s\', 简介:\'%s\'})
                    MERGE (b:花卉品种{name: '%s'})
                    MERGE (a)-[:归属]-> (b)
                    
                    MERGE (c:具体分支{name: '%s'})
                    MERGE (a)-[:属于]-> (c)
                    
                    MERGE (d:%s{name: '%s'})
                    MERGE (a)-[:归于]-> (d)
                ''' % (int(id), name, another_name, img, open_time, desc, class_, belong_to, path1, class_name)
                graph.run(cql)
                for k, b in enumerate(belongs):
                    if k > 0:
                        cql = ''' 
                            MERGE (a:具体分支{name: '%s'})
                            MERGE (b:具体分支{name: '%s'})
                            MERGE (b)-[:从属]-> (a)
                        ''' % (belongs[k-1], belongs[k])
                        graph.run(cql)
            print(flower_path, 'done')
        print(path1, 'done')
    print('花卉构造完成')


if __name__ == '__main__':
    try: 
      test_graph = Graph("http://127.0.0.1:7474/browser/", username="neo4j", password="123456789")
    except ValueError:
      test_graph = Graph("http://127.0.0.1:7474/browser/", auth=("neo4j", "123456789"))
    # test_graph.run('match(n) detach delete n')
    # createEntity0(test_graph)
    # createFlower(test_graph)


'''
CREATE (:国家{id:'1', name:'中国'})	
CREATE (:省份{id:\'1\', name:\'安徽\'})	
CREATE (:产地{id:'1', name:'宣城'})

match (a {name: 'A'}) 
match (b {name: 'B'}) 
CREATE (a)-[r:属于]-> (b)


MATCH (a:`产地`{name:\'%s\'})
MATCH (b:`省份`{name:\'%s\'})
MATCH (c:`国家`{name:\'%s\'})
CREATE (a)-[r:属于]-> (b)
CREATE (b)-[r:属于]-> (c)

match (n) detach delete n	# 删除所有数据
CREATE (n:节点{属性:‘属性值’,....})	# 创建节点
create (n)-[:r{key:'value'}]->(n)	#创建关系
match (n {name: 'value'}) return n	# 以属性值查找节点
match (n: Person) return n			# 以标签查找节点
match (n:movie) where n.released > 1990  return n.title	# 条件查找
'''
