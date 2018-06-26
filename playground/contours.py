# coding=utf-8
"""
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG1 = "../image/contour.png"


def find_contour():
    img = cv2.imread(IMG1)
    blur = cv2.bilateralFilter(img, 5, 65, 56)
    edge = cv2.Canny(blur, 20, 40)
    image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img1 = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imshow("contours", img1)
    cv2.imshow("edge", edge)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("opencv")
    find_contour()