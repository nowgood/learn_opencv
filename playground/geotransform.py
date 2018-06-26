# coding=utf-8
"""
ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
图像的缩放, 平移, 旋转, 仿射变换, 视点变换

scaling:
    cv2.INTER_AREA for shrinking
    cv2.INTER_CUBIC(slow) && cv2.INTER_LINEAR for zooming (放大)

仿射变换
In affine transformation, all parallel lines in the original image
will still be parallel in the output image.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG1 = "../image/saber.jpg"
IMG2 = "../image/tracking.jpeg"


def scaling():
    img = cv2.imread(IMG1)

    # scaling factor
    res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    height, width = img.shape[:2]
    # 注意这里 width, height 在下面函数输入是(宽, 高), shape[:2] 得到 (高, 宽)
    res2 = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_AREA)

    plt.subplot(131)
    plt.imshow(res1)
    plt.title("res1")

    plt.subplot(132)
    plt.imshow(res2)
    plt.title("res2")

    plt.subplot(133)
    plt.imshow(img)
    plt.title("original")

    f = plt.gcf()
    f.set_size_inches(8, 6)
    plt.show()


def translation_rotation():
    img = cv2.imread(IMG1, 0)

    rows, cols = img.shape

    # Third argument of the cv2.warpAffine() function is the size of the output image,
    # which should be in the form of (width, height). Remember
    # width = number of columns, height = number of rows.

    # 平移矩阵数据类型必须是 np.float32
    trans_matrix = np.float32([[1, 0, 20], [0, 1, 80]])
    translation = cv2.warpAffine(img, trans_matrix, (cols, rows))

    # 逆时针旋转80, 扩张 1
    rotation_matrix = cv2.getRotationMatrix2D((cols//2, rows//2), 80, 1)
    rotation = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    plt.subplot(131)
    plt.imshow(translation)
    plt.title("res1")

    plt.subplot(132)
    plt.imshow(rotation)
    plt.title("res2")

    plt.subplot(133)
    plt.imshow(img)
    plt.title("original")

    f = plt.gcf()
    f.set_size_inches(8, 6)
    plt.show()


def affine_transformation():
    """
    In affine transformation, all parallel lines in the original image will still be parallel
    in the output image. To find the transformation matrix,
    we need three points from input image and their corresponding locations in output image.
    """
    img = cv2.imread(IMG1, 0)

    rows, cols = img.shape
    pts1 = np.float32([[10, 10], [10, 50], [50, 10]])
    pts2 = np.float32([[30, 40], [20, 70], [60, 45]])

    # 平移矩阵数据类型必须是 np.float32
    affine_matrix = cv2.getAffineTransform(pts1, pts2)  # 2x3 matrix
    affine = cv2.warpAffine(img, affine_matrix, (cols, rows))

    plt.subplot(122)
    plt.imshow(affine)
    plt.title("affine")

    plt.subplot(121)
    plt.imshow(img)
    plt.title("original")

    f = plt.gcf()
    f.set_size_inches(8, 6)
    plt.show()


def perspective_transformation():

    """
    For perspective transformation, you need a 3x3 transformation matrix.
    Straight lines will remain straight even after the transformation.
    To find this transformation matrix, you need 4 points on the input image
    and corresponding points on the output image.
    Among these 4 points, 3 of them should not be collinear.
    """
    img = cv2.imread(IMG1)
    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols//2, rows//2))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


if __name__ == "__main__":
    # scaling()
    # translation_rotation()
    # affine_transformation()
    perspective_transformation()