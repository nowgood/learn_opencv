# coding=utf-8
"""
For HSV,
    Hue range is [0,179],
    Saturation range is [0,255]
    Value range is [0,255].
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

IMG1 = "../image/saber.jpg"
IMG2 = "../image/tracking.jpeg"


def color_space():
    img = cv2.imread(IMG1)
    if img is None:
        print("can't open image")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("gray", gray), cv2.waitKey(), cv2.destroyAllWindows()

    print("gray shape", gray.shape)

    gray2bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv2rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # cv2.imshow("hsv", hsv), cv2.waitKey(), cv2.destroyAllWindows()

    plt.subplot(321), plt.imshow(img), plt.title("original"), plt.axis("off")
    plt.subplot(322), plt.imshow(img), plt.title("original"), plt.axis("off")
    plt.subplot(323), plt.imshow(gray), plt.title("gray"), plt.axis("off")
    plt.subplot(324), plt.imshow(gray2bgr), plt.title("gray2bgr"), plt.axis("off")
    plt.subplot(325), plt.imshow(hsv), plt.title("hsv"), plt.axis("off")
    plt.subplot(326), plt.imshow(hsv2rgb), plt.title("hsv2rgb"), plt.axis("off")

    plt.show()


def tracking():
    blue = np.uint8([[[255, 0, 0]]])
    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    print("hsv blue", hsv_blue)

    img = cv2.imread(IMG2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("blue target")
    cv2.createTrackbar("small", "blue target", 146, 255, print)
    cv2.createTrackbar("large", "blue target", 255, 255, print)

    while 1:
        pos1 = cv2.getTrackbarPos("small", "blue target")
        pos2 = cv2.getTrackbarPos("large", "blue target")
        lower_bound = np.array([107, 205, pos1])
        upper_bound = np.array([113, 255, pos2])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        blue_obj = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("blue target", blue_obj)
        cv2.imshow("image", img)
        cv2.imshow("mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # color_space()
    tracking()
