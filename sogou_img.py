# -*- coding: utf-8 -*-
import requests
import json
import os
import random
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
def Sougou_GetUrls(num, keyword):
    for i in range((num // 48) + 1):
        page_url = 'https://pic.sogou.com/pics?query=' + keyword + '&mode=1&start={}&reqType=ajax&reqFrom=result&tn=0'.format(i * 48)
        print(page_url)
        page_links_list.append(page_url)
    print(page_links_list)
    return page_links_list

# 初始化锁,创建一把锁
gLock = threading.Lock()


# 生产者，负责从每个页面中获取图片的链接
class Sougou_Producer(threading.Thread):
    def run(self):
        while len(page_links_list) > 0:
            # 上锁
            gLock.acquire()
            # 默认取出列表中的最后一个元素
            page_url = page_links_list.pop()
            # 释放锁
            gLock.release()

            # 获取img标签
            imgs = requests.get(page_url)
            jd = json.loads(imgs.text)
            jd = jd['items']


            # 加锁3
            gLock.acquire()
            for j in jd:
                img_links_list.append(j['pic_url'])
            # 释放锁
            gLock.release()
        print(len(img_links_list))


# 消费者，负责从获取的图片链接中下载图片
class Sougou_Consumer(threading.Thread, ):
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
    Sougou_GetUrls(num, keyword)
    # os.mkdir('./images')
    if os.path.exists('./images'):
        pass
    else:
        os.makedirs('./images')
    # 5个生产者线程，去从页面中爬取图片链接
    for x in range(5):
        Sougou_Producer().start()

    # 10个消费者线程，去从中提取下载链接，然后下载
    for x in range(5):
        Sougou_Consumer().start()

