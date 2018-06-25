# coding=utf-8

"""
画形状的要素：

line 直线 start end
rectangle 矩形 left-top bottom-right
circle 圆 original radius
ellipse 椭圆 original major_axis minor_axis angle(逆时针) start_angle（顺时针） end_angle
polygon 多边形 顶点


thickness : Thickness of the line or circle etc.
If -1 is passed for closed figures like circles,
it will fill the shape. default thickness = 1

thickness = -1 为填充
"""

import cv2
import numpy as np


def cv_draw_shape():
    img = np.zeros([512, 512, 3], dtype=np.uint8)
    cv2.line(img, pt1=(0, 0), pt2=(511, 511), color=(255, 0, 0),
             thickness=3, lineType=cv2.LINE_4)
    cv2.rectangle(img, pt1=(100, 100), pt2=(400, 400), color=(0, 255, 0),
                  thickness=3,  lineType=cv2.LINE_8)
    cv2.circle(img, center=(250, 250), radius=150, color=(0, 0, 255), thickness=3)
    cv2.ellipse(img, center=(250, 250), axes=(150, 80), angle=30, startAngle=90,
                endAngle=310, color=(255, 255, 0), thickness=3)

    points = np.array([[0, 0], [200, 200], [300, 100]], dtype=np.int32)
    points = points.reshape([-1, 1, 2])
    cv2.polylines(img, pts=[points], isClosed=True, color=(0, 255, 255), thickness=3)

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, text="DRAW in CV", org=(60, 490), fontFace=font, fontScale=2,
                color=(255, 255, 240), thickness=3, lineType=cv2.LINE_AA)  # anti-aliased line

    cv2.imshow("draw shapes", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cv_draw_shape()