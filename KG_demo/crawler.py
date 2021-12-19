# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         crawler
# Description:  爬取数据模块
# Author:       Laity
# Date:         2021/10/20
# ---------------------------------------------
import os
from time import sleep
from lxml import etree
import requests, re
from urllib.parse import urljoin, urlencode

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Cookie': '',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
}

base_url = 'http://www.aihuhua.com/hua/'
classification = ['花卉类别', '花卉功能', '应用环境', '盛花期_习性', '养护难度']  # 总类别
page_size = [0, 12, 20, 34, 42, 46]  # 页面范围
flower_class = {}   # 花卉类别
flower_class_file = open('data/种类.txt', 'a', encoding='utf8')

# 界门纲目科属种
Kingdom = {}
Phylum = {}
Class = {}
Order = {}
Family = {}
Genus = {}
Species = {}

Kingdom_file = open('data/科属/界.txt', 'a', encoding='utf8')
Phylum_file = open('data/科属/门.txt', 'a', encoding='utf8')
Class_file = open('data/科属/纲.txt', 'a', encoding='utf8')
Order_file = open('data/科属/目.txt', 'a', encoding='utf8')
Family_file = open('data/科属/科.txt', 'a', encoding='utf8')
Genus_file = open('data/科属/属.txt', 'a', encoding='utf8')
Species_file = open('data/科属/种.txt', 'a', encoding='utf8')
relation_file = open('data/科属/归属关系.txt', 'a', encoding='utf8')

def get_base_html(get):
    if get:
        res = requests.get(base_url, headers=headers)
        file = open('base_html.txt', 'w', encoding='utf8')
        file.write(res.text)
        print('主页源码爬取完成')
    else:
        text = open('base_html.txt', encoding='utf8').read()
        print('主页源码获取完成')
        return text


def get_classification():
    global classification
    base_html = get_base_html(0)
    classification = re.findall('<h2 class="title " title="(.*?)">', base_html)
    content = re.findall('<li><a href="(.*?)" class="a " title="(.*?)" target="_self">', base_html)
    url = 'http://www.aihuhua.com'
    k = 0
    for i in range(len(classification)):
        classification[i] = classification[i].replace(' / ', '_')
        file = open('data/' + '花卉大全.txt', 'a', encoding='utf8')
        # os.mkdir('data/' + classification[i])
        for j in range(page_size[i], page_size[i + 1]):
            open('data/' + classification[i] + '/' + content[j][1] + '.txt', 'w', encoding='utf8')
            file.write(str(k) + '\t' + url + content[j][0] + '\t' + content[j][1] + '\n')
            k += 1
        file.close()
    print('类别构造完成')


def get_content():
    file = open('data/花卉大全.txt', 'r', encoding='utf8')
    lines = file.readlines()
    # page_size = [0, 12, 20, 34, 42, 46]  # 页面范围
    ALL_INDEX = 0
    # 0 6477 10977 20857
    for i in range(0, 5):
        # -------
        # 分批次爬取 修改i的起始值 以及ALL_INDEX的值
        # -------
        for j in range(page_size[i], page_size[i+1]):
            id, url, name = lines[j].split()
            save_file_name = open('data/' + classification[i] + '/' + name + '.txt', 'w', encoding='utf8')
            index = 1
            while True:
                content_page_url = url + 'page-' + str(index) + '.html'
                print(content_page_url, '爬取中...')
                content_page_html = requests.get(content_page_url, headers=headers)
                if content_page_html.status_code == 200:
                    content_page_text = content_page_html.text
                    end = re.findall('class=\'next\'>下一页</a></div>', content_page_text)
                    # 爬取详细花卉信息
                    # ---------------------------------------
                    detail_url = re.findall('<a class="title" target="_blank" title="(.*?)" href="(.*?)">(.*?)</a>', content_page_text)
                    for detail in detail_url:
                        content_url = detail[1]
                        res = requests.get(content_url, headers=headers)
                        if res.status_code == 200:
                            content_text = res.text
                            flower_name = detail[0]
                            anonther_name = re.findall('<label class="cate">别名：(.*?)</label>', content_text)
                            img = re.findall('<img width="140" alt="(.*?)" title="(.*?)" src="(.*?)"', content_text)
                            img_link = img[0][2] if len(img[0]) >=2 and len(img[0][2]) > 0 else '无'
                            flower_class_get = re.findall('<label class="cate">分类：<a href="(.*?)" title="(.*?)" target="_blank">(.*?)</a></label>', content_text)
                            # print(flower_name, flower_class_get[0][-1])
                            if len(flower_class_get) > 0 and flower_class_get[0][-1] not in flower_class:
                                flower_class[flower_class_get[0][-1]] = 1
                                flower_class_file.write(flower_class_get[0][-1] + '\n')
                            belong = re.findall('<label class="cate">科属：(.*?)</label>', content_text)
                            if len(belong[0]) > 0:
                                relation_file.write(str(ALL_INDEX) + '\t' + belong[0] + '\n')
                            belong_line = belong[0].split()
                            for temp in belong_line:
                                if '界' in temp and temp not in Kingdom:
                                    Kingdom[temp] = 1
                                    Kingdom_file.write(temp + '\n')
                                elif '门' in temp and temp not in Phylum:
                                    Phylum[temp] = 1
                                    Phylum_file.write(temp + '\n')
                                elif '纲' in temp and temp not in Class:
                                    Class[temp] = 1
                                    Class_file.write(temp + '\n')
                                elif '目' in temp and temp not in Order:
                                    Order[temp] = 1
                                    Order_file.write(temp + '\n')
                                elif '科' in temp and temp not in Family:
                                    Family[temp] = 1
                                    Family_file.write(temp + '\n')
                                elif '属' in temp and temp not in Genus:
                                    Genus[temp] = 1
                                    Genus_file.write(temp + '\n')
                                elif '种' in temp and temp not in Species:
                                    Species[temp] = 1
                                    Species_file.write(temp + '\n')
                            open_time_get = re.findall('<label class="cate">盛花期：<a title="(.*?)" target="_blank" href="(.*?)">(.*?)</a>', content_text)
                            open_time = '四季'
                            if len(open_time_get) > 0:
                                open_time = open_time_get[0][-1]
                            cmp = re.compile('<p class="desc">(.*?)</p>', re.DOTALL)
                            desc = ' '.join(re.findall(cmp, content_text)[0].split())
                            # print(re.findall(cmp, content_text))
                            # print(desc)
                            # 保存到文件
                            save_text = str(ALL_INDEX) + '\t' + flower_name + '\t'
                            save_text += anonther_name[0] if len(anonther_name[0]) > 0 else '无'
                            save_text += '\t' + img_link
                            save_text += '\t' + flower_class_get[0][-1] if len(flower_class_get) > 0 else '无'
                            save_text += '\t' + belong[0] if len(belong[0]) > 0 else '无'
                            save_text += '\t' + open_time + '\t'
                            save_text += desc if len(desc) > 0 else '无'
                            save_file_name.write(save_text + '\n')
                            ALL_INDEX += 1
                            # print(flower_name, '已保存')
                        else:
                            print(content_url, '爬取失败')
                    # ---------------------------------------
                    if len(end) <= 0:
                        break
                    index += 1
                    # if index == 2:
                    #     break
                else:
                    print(content_page_html, '获取失败')
                # break
            print(name, '爬取完成') # 二级文本
            # sleep(3)
            # break
        print(classification[i], '爬取完成')    # 一级目录
        sleep(10)
        # break
    # 保存分类
    # 保存科属


if __name__ == '__main__':
    get_classification()  # 建立分类文件
    get_content()  # 爬取详细词条
