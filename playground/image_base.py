# coding=utf-8
"""
图片读取， 显示， 保存

opencv 读取图片的格式为 BGR
matplotlib 等其他图片工具读取格式为 RGB
所以 opencv读取使用matplotlib显示， 要使用 cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

1. cv2.imread(img, flag)

    cv2.IMREAD_COLOR (1): Loads a color image. Any transparency of image will be neglected. It is the default flag.
    cv2.IMREAD_GRAYSCALE(0) : Loads image in grayscale mode
    cv2.IMREAD_UNCHANGED(-1) : Loads image as such including alpha ch
    即使图片路径错误也不会报错， 只是返回值为 None

    Use the function cv2.imshow() to display an image in a window.
    The window automatically fits to the image size.

2. cv3.imshow(winname, img)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


GRIL = "../image/loststar.jpg"
ALPHA = "../image/alpha.jpg"
SAVE = "../image/loststar_blur.jpg"


def cv_image():
    flag = 1
    img = cv2.imread(GRIL, flag)
    if img is None:
        print("can not open image")
    pre_img = cv2.medianBlur(img, 3)
    h_img = np.hstack([img, pre_img])

    while 1:
        cv2.namedWindow("gril", cv2.WINDOW_NORMAL)
        cv2.imshow("gril", h_img)
        '''
        If you are using a 64-bit machine, you will have to modify
        k = cv2.waitKey(0) line as follows : k = cv2.waitKey(0) & 0xFF
        '''
        k = cv2.waitKey(2) & 0xFF  # 取低8位

        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            cv2.cv2.imwrite(SAVE, h_img)


def matplotlib_image():
    """
    Color image loaded by OpenCV is in BGR mode.
    But Matplotlib displays in RGB mode.

    So color images will not be displayed
    correctly in Matplotlib if image is read with OpenCV.
    """

    img = plt.imread(GRIL)
    cv_img = cv2.imread(GRIL)

    plt.subplot(121)
    # plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.imshow(img)
    plt.title("matplotlib imread")
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    # plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.imshow(cv_img)
    plt.title("opencv imread")
    plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":
    # cv_image()
    matplotlib_image()