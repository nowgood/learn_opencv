# coding=utf-8
"""
ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html

intensity  强度, [电子]亮度
cv2.add()  是饱和操作, 与 numpy 不同
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG1 = "../image/saber.jpg"
IMG2 = "../image/cv.png"


def arithmetic_op():

    np_img = np.zeros([300, 300, 3], dtype=np.uint8)
    # 创建有类型的变量
    x = np.uint8([255])
    np_img += x  # or np_img[...] = 255

    plt.imshow(np_img, "gray")
    plt.axis("off")

    plt.show()


def blend():
    def nothing(x):
        pass

    img1 = cv2.imread(IMG1)  # BGR
    img2 = cv2.imread(IMG2)  # BGR

    if img1 is None:
        print("can't open image1")
    if img2 is None:
        print("can't open image2")

    img1 = cv2.resize(img1, (300, 400))
    img2 = cv2.resize(img2, (300, 400))
    cv2.namedWindow("blend")
    cv2.createTrackbar("alpha", "blend", 0, 10, nothing)
    while 1:
        alpha = cv2.getTrackbarPos("alpha", "blend") / 10
        blend_img = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
        cv2.imshow("blend", blend_img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


def bit_op():
    # Load two images
    img1 = cv2.imread(IMG1)
    img2 = cv2.imread(IMG2)

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", img2gray)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.inRange(img2gray, 10, 255)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # arithmetic_op()
    # blend()
    bit_op()
