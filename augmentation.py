import cv2
import numpy as np
import albumentations as A

import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('0010.jpg'), cv2.COLOR_BGR2RGB)
#读取图片
(img_height,img_width,img_channels)=img.shape
'''
#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 25
#根据均值和标准差生成符合高斯分布的噪声
gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
#给图片添加高斯噪声
noisy_img = image + gauss
#设置图片添加高斯噪声之后的像素值的范围
noise_img = np.clip(noisy_img,a_min=0,a_max=255)
#保存图片
cv2.imwrite("noisy_img.png",noise_img)
'''
'''
#设置添加椒盐噪声的数目比例
s_vs_p = 0.5
#设置添加噪声图像像素的数目
amount = 0.04
noisy_img = np.copy(image)
#添加salt噪声
num_salt = np.ceil(amount * image.size * s_vs_p)
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
noisy_img[coords[0],coords[1],:] = [255,255,255]
#添加pepper噪声
num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
noisy_img[coords[0],coords[1],:] = [0,0,0]
#保存图片
cv2.imwrite("noisy_img.png",noisy_img)
'''
'''#随机生成一个服从分布的噪声
gauss = np.random.randn(img_height,img_width,img_channels)
#给图片添加speckle噪声
noisy_img = image + image * gauss
#归一化图像的像素值
noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
#保存图片
cv2.imwrite("noisy_img.png",noisy_img)
'''

'''#计算图像像素的分布范围
vals = len(np.unique(image))
vals = 2 ** np.ceil(np.log2(vals))
#给图片添加泊松噪声
noisy_img = np.random.poisson(image * vals) / float(vals)
#保存图片
cv2.imwrite("noisy_img.png",noisy_img)
'''
'''f, ax = plt.subplots(1,2, figsize=(12, 12))
ax[0].imshow(img)
ax[1].imshow(A.FancyPCA(p=1)(image=img)['image'])
plt.show()

cv2.imwrite('FancyPCA.png',A.FancyPCA(p=1)(image=img)['image'])
'''

'''f, ax = plt.subplots(1,2, figsize=(12, 12))
ax[0].imshow(img)
ax[1].imshow(A.ImageCompression(quality_lower=0, p=1)(image=img)['image'])
plt.show()
cv2.imwrite('ImageCompression.png',A.ImageCompression(quality_lower=0, p=1)(image=img)['image'])
'''

f, ax = plt.subplots(1,2, figsize=(12, 12))
ax[0].imshow(img)
cv2.imwrite('RGBShift.png',A.RGBShift(p=1)(image=img)['image'])
ax[1].imshow(A.RGBShift(p=1)(image=img)['image'])
plt.show()


