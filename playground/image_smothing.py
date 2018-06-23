# coding=utf-8
"""
Blur the images with various low pass filters
详情参见 https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

分类:
1. low-pass filters(LPF)
2. high-pass filters(HPF)

作用:
 LPF helps in removing noises, blurring the images etc.
 HPF filters helps in finding edges in the images.

image blur是通过低通滤波器卷积图像实现, 可以有效的移除噪音,
           实际上是从图像中移除了高频信息(noise, edge),
           所以通过这个操作, 边缘也被模糊了一点, 当然这里有blur技术
           不会 blur edge

Gaussian blurring is highly effective in removing gaussian noise from the image.

盐椒噪音 Salt-and-pepper noise

椒盐噪声是指两种噪声，一种是盐噪声（salt noise），另一种是胡椒噪声（pepper noise）。
盐=白色，椒=黑色。前者是高灰度噪声，后者属于低灰度噪声。一般两种噪声同时出现，呈现在图像上就是黑白杂点。

Salt-and-pepper noise is a form of noise sometimes seen on images.
It is also known as impulse noise. This noise can be caused by sharp and sudden disturbances in the image signal.
It presents itself as sparsely occurring white and black pixels.
An effective noise reduction method for this type of noise is a median filter or a morphological filter.

cv2.bilateralFilter() is highly effective in noise removal while keeping edges sharp


双边滤波器:
双边滤波器在空间中也采用高斯滤波器，但是还有一个高斯滤波器是像素差的函数。
空间的高斯函数确保只有附近的像素被考虑用于模糊，而高斯函数的强度差异确保只有
那些与中心像素具有相似强度的像素被认为是模糊的。所以它保留了边缘，因为边缘的像素会有很大的强度变化。
"""

from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

GRID = "/home/wangbin/PycharmProjects/learn_opencv/image/grid.png"
B2W = "/home/wangbin/PycharmProjects/learn_opencv/image/b2w.jpg"
WM = "/home/wangbin/PycharmProjects/learn_opencv/image/watermelon.png"
JTH = "/home/wangbin/PycharmProjects/learn_opencv/image/jiath.jpg"
WATER = "/home/wangbin/PycharmProjects/learn_opencv/image/water.jpg"
BUG = "/home/wangbin/PycharmProjects/learn_opencv/image/bug.jpeg"
FLOW = "/home/wangbin/PycharmProjects/learn_opencv/image/predict.png"
TEXTURE = "/home/wangbin/PycharmProjects/learn_opencv/image/texture.jpg"
BI = "/home/wangbin/PycharmProjects/learn_opencv/image/bilateral.jpeg"


def cv_filter():
    filter_size = 5
    img = cv2.imread(WATER)
    kernel = np.ones([filter_size, filter_size], dtype=np.float)/np.square(filter_size)
    filter_2d = cv2.filter2D(img, -1, kernel)
    box_filter = cv2.boxFilter(img, -1, (3, 3))  # 默认normalize=True
    blur = cv2.blur(img, (3, 3))

    plt.subplot(141)
    plt.imshow(img)
    plt.title("original image")
    plt.xticks([]), plt.yticks([])

    plt.subplot(142)
    plt.imshow(filter_2d)
    plt.title("filter_2d")
    plt.xticks([]), plt.yticks([])

    plt.subplot(143)
    plt.imshow(box_filter)
    plt.title("box_filter")
    plt.xticks([]), plt.yticks([])

    plt.subplot(144)
    plt.imshow(blur)
    plt.title("blur")
    plt.xticks([]), plt.yticks([])

    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    plt.show()


def salt_pepper():
    img = cv2.imread(BUG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)
    blur = cv2.medianBlur(img, 5)  # matplotlib 显示使用 RGB, 而 OpenCV读取为BGR
    gauss = cv2.GaussianBlur(img, (5, 5), 0)

    plt.subplot(131)
    plt.imshow(img)
    plt.title("original image")
    plt.xticks([]), plt.yticks([])

    plt.subplot(132)
    plt.imshow(gauss)
    plt.title("Gauss blur")
    plt.xticks([]), plt.yticks([])

    plt.subplot(133)
    plt.imshow(blur)
    plt.title("Median blur")
    plt.xticks([]), plt.yticks([])

    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.show()


def cv_bilateral():
    img = cv2.imread(BI)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k_size = 5
    print(img.shape)
    blur = cv2.medianBlur(img, k_size)  # matplotlib 显示使用 RGB, 而 OpenCV读取为BGR
    gauss = cv2.GaussianBlur(img, (k_size, k_size), 0)
    bilateral = cv2.bilateralFilter(img, k_size, 15, 10)

    plt.subplot(141)
    plt.imshow(img)
    plt.title("original image")
    plt.xticks([]), plt.yticks([])

    plt.subplot(142)
    plt.imshow(gauss)
    plt.title("Gauss blur")
    plt.xticks([]), plt.yticks([])

    plt.subplot(143)
    plt.imshow(blur)
    plt.title("Median blur")
    plt.xticks([]), plt.yticks([])

    plt.subplot(144)
    plt.imshow(bilateral)
    plt.title("bilateral blur")
    plt.xticks([]), plt.yticks([])

    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.show()


if __name__ == "__main__":
    cv_filter()
    # salt_pepper()
    # cv_bilateral()