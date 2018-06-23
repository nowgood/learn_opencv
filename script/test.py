# coding=utf-8
import cv2
import math
import numpy as np

PNG = "/home/wangbin/PycharmProjects/learn_opencv/predict.png"


def canny_binary():
    one = np.arange(16, dtype=np.uint8).reshape(4, 4)
    edge = cv2.Canny(one, 10, 3)
    print(edge)
    mask = np.where(edge > 0, 1, 0)
    print(mask)


def numpy_func():
    n = np.ones([2, 2])
    sn = np.stack(n, axis=-1)
    print(sn)


def read_image():
    src = cv2.imread(PNG, 0)
    src_rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    print("gray shape", src.shape)
    print("color shape", src_rgb.shape)
    cv2.imshow("input", src)
    cv2.imshow("output", src_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread(PNG, 1)
    one = np.arange(4, dtype=np.uint8).reshape(2, 2)
    one_ = cv2.cvtColor(one, cv2.COLOR_GRAY2BGR)
    print(one)
    print(one_)
    print(one.shape, one_.shape)