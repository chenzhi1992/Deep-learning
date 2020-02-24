# -*- coding: utf-8 -*-
import requests
import json
import os
import random
from lxml import etree
import re
import socket
import lxml
import threading
# 设置请求超时时间，防止长时间停留在同一个请求
socket.setdefaulttimeout(10)

# 页面链接的初始化列表
page_links_list = []
# 图片链接列表
img_links_list = []
# 获取爬取的页数和页面链接
def Biying_GetUrls(num, keyword):
    for i in range((num // 35) + 1):
        page_url = 'https://cn.bing.com/images/async?q={}&first={}&count=35&relo=4&relp=5&cw=1117&ch=689&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1&dgState=c*6_y*1582s1599s1589s1660s1720s1704_i*40_w*172&IG=B3C2B933EAED48A4A82330EC1E7A638B&SFX=2&iid=images.5659'.format(keyword, i * 35)
        print(page_url)
        page_links_list.append(page_url)
    print(page_links_list)
    return page_links_list

# 初始化锁,创建一把锁
gLock = threading.Lock()


# 生产者，负责从每个页面中获取图片的链接
class Biying_Producer(threading.Thread):
    def run(self):
        while len(page_links_list) > 0:
            # 上锁
            gLock.acquire()
            # 默认取出列表中的最后一个元素
            page_url = page_links_list.pop()
            # 释放锁
            gLock.release()

            # 获取img标签
            html = requests.get(page_url).text
            html = etree.HTML(html)
            conda_list = html.xpath('//a[@class="iusc"]/@m')

            # 加锁3
            gLock.acquire()

            for j in conda_list:
                img_url = re.search('"murl":"(.*?)"', j).group(1)
                img_links_list.append(img_url)
            # 释放锁
            gLock.release()
        print(len(img_links_list))


# 消费者，负责从获取的图片链接中下载图片
class Biying_Consumer(threading.Thread, ):
    def run(self):
        print("%s is running" % threading.current_thread())
        while True:
            # print(len(img_links_list))
            # 上锁
            gLock.acquire()
            if len(img_links_list) == 0:
                # 不管什么情况，都要释放锁
                gLock.release()
                continue
            else:
                img_url = img_links_list.pop()
                print(img_links_list)
                gLock.release()
                filename = img_url.split('/')[-1]
                print('正在下载：', filename)
                path = './images/' + filename
                try:
                    with open(path, 'wb+') as f:
                        f.write(requests.get(img_url).content)
                except:
                    continue
                if len(img_links_list) == 0:
                    exit()


if __name__ == '__main__':
    keyword = '口罩'
    num = int(input('请输入爬取图片数目：'))
    Biying_GetUrls(num, keyword)
    # os.mkdir('./images')
    if os.path.exists('./images'):
        pass
    else:
        os.makedirs('./images')
    # 5个生产者线程，去从页面中爬取图片链接
    for x in range(5):
        Biying_Producer().start()

    # 10个消费者线程，去从中提取下载链接，然后下载
    for x in range(5):
        Biying_Consumer().start()
