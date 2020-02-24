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
def Baidu_GetUrls(num, keyword):
    for i in range((num // 30) + 1):
        page_url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word={}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=girl&pn={}&rn=30&gsm=1e&1581069586398='.format(keyword, keyword, 30 * i)
        print(page_url)
        page_links_list.append(page_url)
    print(page_links_list)
    return page_links_list

# 初始化锁,创建一把锁
gLock = threading.Lock()


# 生产者，负责从每个页面中获取图片的链接
class Baidu_Producer(threading.Thread):
    def run(self):
        while len(page_links_list) > 0:
            # 上锁
            gLock.acquire()
            # 默认取出列表中的最后一个元素
            page_url = page_links_list.pop()
            # 释放锁
            gLock.release()

            # 获取img标签
            r = requests.get(page_url).text
            res = json.loads(r)['data']

            # 加锁3
            gLock.acquire()

            if res:
                print(res)
                for j in res:
                    try:
                        url = j['middleURL']
                        img_links_list.append(url)
                    except:
                        print('该图片的url不存在')
            # 释放锁
            gLock.release()
        print(len(img_links_list))


# 消费者，负责从获取的图片链接中下载图片
class Baidu_Consumer(threading.Thread, ):
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
    keyword = '性感美女'
    num = int(input('请输入爬取图片数目：'))
    Baidu_GetUrls(num, keyword)
    # os.mkdir('./images')
    if os.path.exists('./images'):
        pass
    else:
        os.makedirs('./images')
    # 5个生产者线程，去从页面中爬取图片链接
    for x in range(5):
        Baidu_Producer().start()

    # 10个消费者线程，去从中提取下载链接，然后下载
    for x in range(5):
        Baidu_Consumer().start()
