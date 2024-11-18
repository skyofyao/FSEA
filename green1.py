# -*- coding: utf-8 -*- 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 使用2g-r-b分离土壤与背景
 
src = cv2.imread('few-shot/33653.jpg')
#cv2.imshow('src', src)
 
# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b,g,r) = cv2.split(fsrc)

print(r.shape)

gray = 2 * g - b - r
 
# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
 
# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
#plt.plot(hist)
#plt.show()
#cv2.waitKey()
 
# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1, 255, cv2.THRESH_OTSU)
# plt.savefig("C:/Users/Admin/Desktop/1.jpg")
#cv2.imwrite('D:/maskimg/00536.jpg',bin_img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
#cv2.imshow('bin_img', bin_img)
#cv2.waitKey()
# 得到彩色的图像
#cv2.imshow('img',bin_img)
#cv2.imwrite('mask2/1111.jpg',bin_img)
#cv2.waitKey()
(b8, g8, r8) = cv2.split(src)
color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
cv2.imwrite('Mask1/33653.jpg',color_img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
#cv2.imwrite('D:/学习资料重要/检测/目标检测/代码/faster-rcnn-pytorch-master/2.jpg',color_img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
#cv2.imshow('color_img', color_img)
#cv2.waitKey()