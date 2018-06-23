# coding=utf-8

from __future__ import print_function
import cv2
import numpy as np
import tempfile
from math import ceil
import caffe


VIDEO = "/home/wangbin/PycharmProjects/learn_opencv/video/driving.mp4"
caffemodel = "/home/wangbin/github/flownet2/models/FlowNet2-KITTI/FlowNet2-KITTI_weights.caffemodel.h5"
deployproto = "/home/wangbin/github/flownet2/models/FlowNet2-KITTI/FlowNet2-KITTI_deploy.prototxt.template"
WRITER = "/home/wangbin/PycharmProjects/learn_opencv/video/frame_skip_4_flow.avi"
SKIP = 2


def video2flow():
    cap = cv2.VideoCapture(VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    fps = 2.0
    writer = cv2.VideoWriter(WRITER, fourcc, fps, (width/2, height))

    # _, frame1 = cap.read()
    # _, frame2 = cap.read()
    # print("frame1 dtype", frame1.dtype)

    net = load_model(width, height)
    while True:
        # get a frame
        # frame1 = frame2
        # _, frame2 = cap.read()
        i = 0
        _, frame1 = cap.read()
        while i < SKIP:
            cap.read()
            i += 1
        _, frame2 = cap.read()
        if frame2 is None:
            break
        blob = predict_flow(frame1, frame2, net)
        flow = visualize_optical_flow(frame1, blob)
        hmerge = np.vstack((frame1, flow))
        hmerge = cv2.resize(hmerge, (width/2, height))
        cv2.imshow("frame-flow", hmerge)
        writer.write(hmerge)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def load_model(width, height):
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width / divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height / divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH'])
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    proto = open(deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    caffe.set_logging_disabled()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, caffemodel, caffe.TEST)
    print('Network forward pass using %s.' % caffemodel)

    return net


def predict_flow(frame0, frame1, net):

    num_blobs = 2
    input_data = [frame0[np.newaxis, :, :, :].transpose(0, 3, 1, 2),
                  frame1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)]  # batch, bgr, h, w

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    # There is some non-deterministic nan-bug in caffe
    i = 1
    while i <= 5:
        i += 1
        net.forward(**input_dict)
        contains_NaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                contains_NaN = True

        if not contains_NaN:
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)  # CHW -> HWC
    return blob


def visualize_optical_flow(frame1, blob):
    # optical flow visualization 光流可视化
    # 由于frame的数据类型为np.uint8 即 usigned char, 最大存储值为255, 如果赋值为256, 结果为 0,
    # 也就是说及时赋值很大, 也会被截断
    # 对于 饱和度s 和亮度v 而言, 最大值是255, s = 255 色相最饱和, v = 255, 图片最亮
    # 而对与颜色而言, opencv3中, (0, 180) 就会把所有颜色域过一遍, 所以这里下面就算角度时会除以 2

    # np.zeros_like(): Return an array of zeros with the same shape and type as a given array.
    hsv = np.zeros_like(frame1)

    # cv2.cartToPolar(x, y): brief Calculates the magnitude and angle of 2D vectors.
    mag, ang = cv2.cartToPolar(blob[..., 0], blob[..., 1])

    # degree to rad: degree*180/np.pi
    hsv[..., 0] = (ang * 180 / np.pi) / 2

    # brief Normalizes the norm or value range of an array
    # norm_type = cv2.NORM_MINMAX, 即将值标准化到(0, 255)
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 亮度为255
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == "__main__":
    print("start predict optical flow")
    video2flow()