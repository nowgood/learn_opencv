# coding=utf-8

"""
参考： http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


def web_camera():

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def rstp_video():

    cap = cv2.VideoCapture("rtsp://184.72.239.149/vod/mp4://BigBuckBunny_175k.mov")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            print("video arrives end!")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def video_property():
    """
     You can also access some of the features of this video using cap.get(propId) method where propId is a number from 0 to 18.
     Each number denotes a property of the video (if it is applicable to that video)

     Some of these values can be modified using cap.set(propId, value). Value is the new value you want.
    """
    cap = cv2.VideoCapture("rtsp://184.72.239.149/vod/mp4://BigBuckBunny_175k.mov")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    print("width height fps count")
    print(width, height, fps, frame_count)

    cap.release()


def video_writer():
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 15
    writer = cv2.VideoWriter("../video/cam.avi", fourcc, fps, (640, 480))
    while True:
        _, frame = cap.read()
        # frame = cv2.flip(frame, 0)  翻转
        writer.write(frame)
        cv2.imshow("cam", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("viedo base\n")
    # web_camera()
    # rstp_video()
    # video_property()
    video_writer()