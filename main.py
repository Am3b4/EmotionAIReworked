import multiprocessing
import traceback
import cProfile
import pstats
import keras
import cv2


# snakeviz ./STATS.prof

def main():

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = keras.saving.load_model('Models/LiteEmotionAI_7.keras')
    vid = cv2.VideoCapture(0)

    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except (KeyboardInterrupt, OverflowError):
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
