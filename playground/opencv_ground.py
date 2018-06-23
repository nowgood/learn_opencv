# coding=utf-8
from __future__ import print_function
import cv2
import numpy as np

VIDEO = "/home/wangbin/PycharmProjects/learn_opencv/midpredict_KITTI.avi"
WRITER = "/home/wangbin/PycharmProjects/learn_opencv/video/edge_KITTI_no_blur_50_25.avi"
TEMP = "/home/wangbin/PycharmProjects/learn_opencv/video/edge_temp.avi"
PNG = "/home/wangbin/PycharmProjects/learn_opencv/predict.png"
GRID = "/home/wangbin/PycharmProjects/learn_opencv/image/grid.png"
B2W = "/home/wangbin/PycharmProjects/learn_opencv/image/b2w.jpg"
WM = "/home/wangbin/PycharmProjects/learn_opencv/image/watermelon.png"

def video2flow():
    cap = cv2.VideoCapture(VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    fps = 8.0
    writer = cv2.VideoWriter(TEMP, fourcc, fps, (width, height))

    while True:
        # get a frame
        _, frame = cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur_edge = cv2.blur(frame, (3, 3))
        # blur = cv2.GaussianBlur(frame, (5, 5), 0)
        edge = cv2.Canny(gray, 50, 25)  # Canny算子处理之后的是灰度图, 单通道
        mask = np.where(edge > 0, 1, 0).astype(np.uint8)  # 获取掩码, 图像的数据类型为 unsigned char
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 单通道转3通道
        edge_bgr = np.multiply(mask, frame)

        cv2.imshow("original frame", frame)
        # cv2.imshow("blur", blur)
        cv2.imshow("canny edge detect", edge_bgr)

        writer.write(edge_bgr)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def cv_inRange():
    arr = np.arange(5)
    low = np.array(2)
    upper = np.array(4)
    mask = cv2.inRange(arr, low, upper)
    cv2.bitwise_and(arr, arr, mask=mask)
    print(mask)
    print(arr)


def image_gradient():
    img = cv2.imread(B2W)
    img = cv2.medianBlur(img, ksize=5)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    sobel_uint8_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_uint8_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    laplace = cv2.Laplacian(img, cv2.CV_64F)
    laplace_uint8 = cv2.Laplacian(img, cv2.CV_8U)

    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_uint8_x", sobel_uint8_x)
    print(sobel_x)
    print(sobel_uint8_x)
    # cv2.imshow("sobel_y", sobel_y)
    # cv2.imshow("laplace", laplace)
    # cv2.imshow("sobel_y", sobel_y)
    # cv2.imshow("laplace", laplace)
    # cv2.imshow("laplace_uint8", laplace_uint8)
    edge = cv2.Canny(img, 30, 15)
    cv2.imshow("edge", edge)

    cv2.waitKey()
    cv2.destroyAllWindows()


def edge_detect():
    # 使用 Canny 算子之前进行LPF滤波, 很有必要
    img = cv2.imread(WM)
    edge = cv2.Canny(img, 50, 20)
    med_blur_img = cv2.medianBlur(img, 7)
    gauss_blur_img = cv2.GaussianBlur(img, (7, 7), 0)
    med_blur_edge = cv2.Canny(med_blur_img, 50, 20)
    gauss_blur_edge = cv2.Canny(gauss_blur_img, 50, 20)
    kernel = np.ones([7, 7], dtype=np.uint8)
    morph_edge = cv2.morphologyEx(gauss_blur_edge, cv2.MORPH_CLOSE, kernel=kernel)
    cv2.imshow("edge", edge)
    cv2.imshow("med_blur_edge", med_blur_edge)
    cv2.imshow("gauss_blur_edge", gauss_blur_edge)
    cv2.imshow("morph_edge", morph_edge)             
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # video2flow()
    # cv_inRange()
    # image_gradient()
    edge_detect()