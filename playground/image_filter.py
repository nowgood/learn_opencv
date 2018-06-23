# coding=utf-8
"""
 Image Filtering
 详情参见 https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html

 Morphological Operations 形态运算符
 一组基于形状进行图像处理的操作, 形态运算符将结构化元素应用于输入图像并生成输出图像。

 RGB白色 #FFF

"""

from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

ERODE = "/home/wangbin/PycharmProjects/learn_opencv/image/erode.png"


def cv_dilate():
    filter_size = 3
    img = cv2.imread(ERODE)
    kernel = np.ones((filter_size, filter_size), dtype=np.uint8)
    dilate = cv2.dilate(img, kernel)
    erode_after_dilate = cv2.erode(dilate, kernel)

    plt.subplot(131)
    plt.imshow(img)
    plt.title("original image")
    plt.xticks([]), plt.yticks([])

    # plt.subplot(132)
    # plt.imshow(dilate)
    # plt.title("dilate image")
    # plt.xticks([]), plt.yticks([])

    plt.subplot(133)
    plt.imshow(erode_after_dilate)
    plt.title("erode after dilate")
    plt.xticks([]), plt.yticks([])

    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.show()


def cv_erode():
    filter_size = 3
    img = cv2.imread(ERODE)
    kernel = np.ones((filter_size, filter_size), dtype=np.uint8)
    erode = cv2.erode(img, kernel, iterations=1)
    dilate_afer_erode = cv2.dilate(erode, kernel, iterations=1)

    plt.subplot(131)
    plt.imshow(img)
    plt.title("original image")

    # plt.subplot(132)
    # plt.imshow(erode)
    # plt.title("erode image")

    plt.subplot(133)
    plt.imshow(dilate_afer_erode)
    plt.title("dilate after erode")

    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.show()


def cv_opening_closing():
    filter_size = 3
    img = cv2.imread(ERODE)
    # 获取矩形, 椭圆形等形状的 filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #  use erosion after dilation = morphological closing
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    plt.subplot(221)
    plt.imshow(img)
    plt.title("original image")

    plt.subplot(222)
    plt.imshow(opening)
    plt.title("opening image")

    plt.subplot(223)
    plt.imshow(closing)
    plt.title("closing image")

    plt.subplot(224)
    plt.imshow(gradient)
    plt.title("gradient image")

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()


if __name__ == "__main__":
    # cv_dilate()
    # cv_erode()
    cv_opening_closing()
