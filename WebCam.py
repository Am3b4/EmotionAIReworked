import multiprocessing
import cv2


def getImage(vid: cv2.VideoCapture, queue: multiprocessing.Queue) -> None:

    """

    :param vid: cv2.Videocapture() Object; reads the image from the video camera
    :param queue: Multiprocessing Queue Object; the image is placed inside this queue
    """

    ret, img = vid.read()

    if ret:
        queue.put(cv2.flip(img, 1))
