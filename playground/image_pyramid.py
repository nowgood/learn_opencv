# coding=utf-8
"""
ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html#pyramids
图像金字塔 image pyramids

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG1 = "../image/messi5.jpg"


def pyramids():
    img = cv2.imread(IMG1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    low_reso1 = cv2.pyrDown(img)
    low_reso2 = cv2.pyrDown(low_reso1)
    low_reso3 = cv2.pyrDown(low_reso2)

    plt.subplot(221), plt.imshow(img), plt.title("original")
    plt.subplot(222), plt.imshow(low_reso1), plt.title("low reso1")
    plt.subplot(223), plt.imshow(low_reso2), plt.title("low_reso2")
    plt.subplot(224), plt.imshow(low_reso3), plt.title("low_ reso3")
    plt.show()


if __name__ == "__main__":
    print("opencv")
    pyramids()