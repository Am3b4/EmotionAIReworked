import multiprocessing
import time
import cv2


def getImage(queue: multiprocessing.Queue) -> None:

    """
    :param queue: Multiprocessing Queue Object; the image and the start time are placed inside this queue
    """

    while True:

        start = time.perf_counter()

        vid = cv2.VideoCapture(0)
        ret, img = vid.read()

        if ret:
            img = cv2.flip(img, 1)

            print(f'{start}')

            cv2.imshow('test', img)

            queue.put([img, start])
