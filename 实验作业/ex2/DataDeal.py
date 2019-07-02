
#coding:utf-8
import os
from PIL import Image
import numpy as np
'''
Python图像处理库PIL的基本概念介绍”，我们知道PIL中有九种不同模式。
分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F。
模式“1”为二值图像，非黑即白。但是它每个像素用8个bit表示，0表示黑，255表示白
模式“L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的。
'''
def fun(src,dst):
    if not os.path.exists(src):
        return
    if not os.path.exists(dst):
        os.mkdir(dst)
    list=os.listdir(src)
    length=len(list)
    for i in range(length):
        path=src+'/'+list[i]
        SavePath=dst+'/'+list[i][:-4]+".txt"
        read=Image.open(path).convert("1")
        arr=np.asarray(read)
        np.savetxt(SavePath,arr,fmt="%d",delimiter='')    #保存格式为整数,没有间隔
        #np.savetxt(SavePath, arr,fmt="%d")

#训练集
src="F:/program/Machine Learning/src"
dst="F:/program/Machine Learning/dst"


#  #测试集
# src="F:/program/Machine Learning/src1"
# dst="F:/program/Machine Learning/dst1"
fun(src,dst)