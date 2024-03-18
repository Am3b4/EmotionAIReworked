from AnalyzeImage import analyzeImage
from WebCam import getImage

import multiprocessing
import traceback
import cProfile
import pstats
import keras
import cv2


# snakeviz ./STATS.prof

def main():

    queue1 = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=getImage, args=(queue1,))
    p2 = multiprocessing.Process(target=analyzeImage, args=(queue1,))

    # Start processes
    p1.start()
    p2.start()


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except:
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
