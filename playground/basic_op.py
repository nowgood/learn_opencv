# coding=utf-8
"""
intensity  强度, [电子]亮度

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG1 = "../image/saber.jpg"


def basic_op():
    img = cv2.imread(IMG1)  # BGR
    if img is None:
        print("can't open image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel = img[30, 30]
    blue = img[30, 30, 0]
    img[100:104, 206:210] = [255, 0, 0]
    print("BGR: ", pixel, blue)
    print("shape(HxW) ", img.shape)
    print("size ", img.size)
    print("date type ", img.dtype)

    # ROI: region of Image
    eye = img[105:116, 165:186]
    # cv2.rectangle(img, (185, 115), (165, width5), (255, 0, 0), thickness=2)
    img[75:86, 170:191] = eye
    if img[170:191, 75:86].all() == eye.all():
        print("ses")
    plt.imshow(img, "gray")
    plt.axis("off")
    # b, g, r = cv2.split(img)
    # nb, ng, nr = img[..., 0], img[..., 1], img[..., 2]
    #
    # plt.subplot(321), plt.imshow(b), plt.title('b'), plt.axis("off")
    # plt.subplot(323), plt.imshow(g), plt.title('g'), plt.axis("off")
    # plt.subplot(325), plt.imshow(r), plt.title('r'), plt.axis("off")
    #
    # plt.subplot(322), plt.imshow(nb), plt.axis("off")
    # plt.subplot(324), plt.imshow(ng), plt.axis("off")
    # plt.subplot(326), plt.imshow(nr), plt.axis("off")

    plt.show()


def border():
    blue = [0, 0, 255]  # matplotlib
    width = 30
    img1 = cv2.imread(IMG1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    replicate = cv2.copyMakeBorder(img1, width, width, width, width, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, width, width, width, width, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, width, width, width, width, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, width, width, width, width, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img1, width, width, width, width, cv2.BORDER_CONSTANT, value=blue)

    plt.subplot(231), plt.imshow(img1, "gray"), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_width1')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

    f = plt.gcf()
    f.set_size_inches(15, width)
    plt.show()


if __name__ == "__main__":
    # basic_op()
    border()
