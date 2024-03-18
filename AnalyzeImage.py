import multiprocessing
import keras
import numpy
import time
import cv2


def analyzeImage(queueIn: multiprocessing.Queue):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = keras.saving.load_model('Models/LiteEmotionAI_7.keras')

    while True:

        faceList = []
        faceListCord = []

        try:

            img, start = queueIn.get()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)

            faceList, faceListCord, img = getFaces(faces, faceList, faceListCord, img)

            if faceList:

                predictions = detectEmotion(faceList, model)
                img = evaluatePredictions(predictions, img, faceListCord)

            img = fps(img, start)

            cv2.imshow('WebCam', img)

        except multiprocessing.Queue.empty:
            pass


def fps(img, start):

    end_t = time.perf_counter()
    dif_t = end_t - start

    FPS = int(1 / dif_t)

    img = cv2.putText(img, f'FPS: {FPS}', (0, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                      1, cv2.LINE_AA, False)
    print(f'FPS: {FPS}')

    return img


def evaluatePredictions(predictions, img, faceListCord):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    for i, prediction in enumerate(predictions):

        org = (faceListCord[i][0], faceListCord[i][1])
        score = float(keras.ops.sigmoid(prediction))

        happy = f'{100 * (1 - score):.2f}'
        not_happy = f'{100 * score:.2f}'

        if happy >= not_happy:
            img = cv2.putText(img, f'Happy: {happy}%', org, font, fontScale,
                              (0, 255, 0), thickness, cv2.LINE_AA, False)
        else:
            img = cv2.putText(img, f'Not Happy: {not_happy}%', org, font, fontScale,
                              (0, 0, 255), thickness, cv2.LINE_AA, False)

    return img


def detectEmotion(faceList, model: keras.Model):

    faceArray = numpy.asarray(faceList)
    return model.predict(faceArray)


def getFaces(faces, faceList, faceListCord, img):

    for i, (x, y, w, h) in enumerate(faces):

        mult = 1.2
        x, y, w, h = multCoord(x, y, w, h, mult)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))

        faceList.append(face)
        faceListCord.append((x, y, w, h))

    return faceList, faceListCord, img


def multCoord(x: int, y: int, w: int, h: int, mult: float) -> tuple[int, int, int, int]:

    """
    The functions takes as input the coordinates of the face and makes the window size bigger.
    """

    x = int(x / mult)
    y = int(y / mult)
    w = int(w * mult)
    h = int(h * mult)

    return x, y, w, h
