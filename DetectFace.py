import multiprocessing
import cv2


def detectFace(queueIn: multiprocessing.Queue, queueOut: multiprocessing.Queue, face_cascade: cv2.CascadeClassifier):

    while True:

        faceList = []
        faceListCord = []

        img = queueIn.get()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):

            mult = 1.2
            x, y, w, h = multCoord(x, y, w, h, mult)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))

            faceList.append(face)
            faceListCord.append((x, y, w, h))


def multCoord(x: int, y: int, w: int, h: int, mult: float) -> tuple[int, int, int, int]:

    x = int(x / mult)
    y = int(y / mult)
    w = int(w * mult)
    h = int(h * mult)

    return x, y, w, h
