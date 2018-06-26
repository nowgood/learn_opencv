# coding=utf-8
"""
ipython:
    os.getcwd()

    cv2.setUseOptimized(False)
    print(cv2.useOptimized())
    %timeit cv2.bilateralFilter(img, 5, 74, 74)
    cv2.setUseOptimized(True)
    %timeit cv2.bilateralFilter(img, 5, 74, 74)
"""

import cv2

IMG1 = "../image/saber.jpg"


def permance():
    img = cv2.imread(IMG1, 0)
    e1 = cv2.getTickCount()
    # 双边滤波的时间消耗有点大
    # img = cv2.bilateralFilter(img, 3, 75, 75)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edge = cv2.Canny(img, 30, 50)
    e2 = cv2.getTickCount()

    t = (e2 - e1) / cv2.getTickFrequency()
    print("blur Canny time: ", t)
    cv2.imshow("edge", edge)
    cv2.waitKey()
    cv2.destroyAllWindows()


def use_optimal_code():
    print(cv2.useOptimized())


if __name__ == "__main__":
    # permance()
    use_optimal_code()