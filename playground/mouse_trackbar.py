import cv2
import numpy as np

GRIL = "../image/loststar.jpg"
DANCE = "../image/dance.jpeg"


def show_event():
    events = [i for i in dir(cv2) if 'EVENT' in i]
    for i in range(1, len(events)+1):
        print(events[i - 1], end='\t')
        if i % 3 == 0:
            print()


def mouse_draw_circle():
    # mouse callback function
    def draw_circle(event, x, y, flags, param):  # 必须是传入5个参数， 负责有 warning
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 100, (255, 255, 0), -1)
        print(flags)

    # Create a black image, a window and bind the function to window
    img = np.zeros((512, 512, 3), np.uint8)
    img[...] = 255

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw_circle)

    while True:
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def mouse_draw_rect():

    def draw_rect(event, x, y, flags, param):
        nonlocal mode, drawing, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if mode:
                    cv2.rectangle(img, (ix, iy), (x, y), color=(255, 0, 0), thickness=3)
                else:
                    cv2.circle(img, (x, y), 30, color=(0, 255, 0), thickness=4)

    mode = False
    drawing = False
    ix, iy = 0, 0
    img = np.ones([500, 500, 3], dtype=np.uint8)
    img[...] = 255
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", draw_rect)
    while True:
        cv2.imshow("image", img)
        k = cv2.waitKey(2) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            mode = not mode

    cv2.destroyAllWindows()


def track_bar():
    def on_tracker(size):
        nonlocal img
        blur = cv2.medianBlur(img, size)
        edge = cv2.Canny(blur, )
        cv2.imshow("blur", blur)

    img = cv2.imread(GRIL)
    cv2.namedWindow("blur")
    cv2.createTrackbar("sliding", "blur", 0, 9, on_tracker)
    cv2.imshow("blur", img)
    while True:
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def two_track_bar(img_path):
    def on_tracker(x):
        pass

    img = cv2.imread(img_path)
    if img is None:
        print("can't open image")
    cv2.namedWindow("edge")
    cv2.createTrackbar("threshold1", "edge", 0, 150, on_tracker)
    cv2.createTrackbar("threshold2", "edge", 0, 75, on_tracker)
    while True:
        t1 = cv2.getTrackbarPos("threshold1", "edge")
        t2 = cv2.getTrackbarPos("threshold2", "edge")
        edge = cv2.Canny(img, t1, t2)
        mask = cv2.inRange(edge, 1, 255)
        bgr_edge = cv2.bitwise_and(img, img, mask=mask)
        h = np.hstack([img, bgr_edge])
        cv2.imshow("edge", h)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # show_event()
    # mouse_draw_circle()
    # mouse_draw_rect()
    # track_bar()
    two_track_bar(DANCE)