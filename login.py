# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:36:57 2017

@author: ARH
"""

import requests
import urllib.request
from requests.auth import HTTPProxyAuth
#proxyDict = {'http' : '600','https':'600'}
login_data = dict(mode=191,username='abey2015cse',password='abey123',producttype=0)
auth = HTTPProxyAuth('foss','foss@ajc')
r=requests.get("http://192.168.0.1:8090/login.xml",proxies=proxyDict,auth=auth)
with requests.Session() as c:
    proxy_support=urllib.request.ProxyHandler({"http":"http://192.168.0.1:8090/login.xml"})
    opener=urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
    url='http://192.168.0.1:8090/login.xml'
    c.get(url)
    c.post(url,data=login_data,header={"Referer":"http://192.168.0.1:8090/login.xml"})
    page = c.get('http://192.168.0.1:8090/login.xml')
    print (page.content)