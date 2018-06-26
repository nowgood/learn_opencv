# coding=utf-8
"""
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
IMG1 = "../image/corner.jpeg"


def find_corner():

    img = cv2.imread(IMG1)
    if img is None:
        print("can't open image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 3, 5, 0.1)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, np.ones([3, 3]))

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("opencv")
    find_corner()