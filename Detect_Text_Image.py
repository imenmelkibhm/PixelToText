#This file includes the core function for text detection in a given image

import sys
import os
import argparse
import numpy as np
import cv2
import time
import logging
from ctypes import *
from PIL import Image, ImageEnhance

#Define logging level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def er_draw():
    return 0


def text_recognition(img, Debug,i):
    #Preprocessing to enhance text structure in the image
    #Image enhancement using PIL
    image = Image.fromarray(img)
    contrast = ImageEnhance.Contrast(image)
    contrasted = contrast.enhance(1)
    #contrasted.show()
    img = np.array(contrasted)
    (rows, cols) = (img.shape[0], img.shape[1])

    # Load the dll
    mydll = cdll.LoadLibrary("TextMSER/Text.so")
    mydll.text_recognition(img.ctypes.data_as(POINTER(c_ubyte)), rows, cols,i, Debug)

def text_detect_image(img, Debug):

    pathname = os.path.dirname(os.path.abspath(sys.argv[0]))
    # for visualization
    vis = img.copy()
    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))

    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    #print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
    #print("    (...) this may take a while (...)")
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1(pathname + '/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2(pathname + '/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)

        regions = cv2.text.detectRegions(channel, er1, er2)
        #rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
        rects, regions_groupes = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,pathname + '/trained_classifier_erGrouping.xml',0.5)
        #Visualization
        for r in range(0,np.shape(rects)[0]):
            rect = rects[r]
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 255), 1)

    #Visualization
    if Debug:
        cv2.imshow("Text detection result", vis)
        cv2.waitKey(0)


    #Text Recognition OCR
    out_img=img.copy()
    out_img_detection = img.copy()
    out_img_segmentation = np.zeros(img.shape[0], img.shape[1], np.uint8)
    scale_img = 600/float(img.shape[0])
    scale_font = (float)(2-scale_img)/1.4


    for i in xrange(0,np.shape(rects)[0]):
        rect = rects[i]
        cv2.rectangle(out_img_detection, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 255), 3 )
        #group_img = np.zeros(img.shape[0], img.shape[1], np.uint8)



    return rects

#the main function if used standalone
def main():
    #Read input arguments
    parser = argparse.ArgumentParser(description='Detect and recognize the text present in the input image')
    parser.add_argument("-i", "--input", help="input image file")
    parser.add_argument("-o", "--outputpath", help="the path to save the output files")
    parser.add_argument("-n", "--outputname", help="the name to save the output files")
    parser.add_argument("-d", "--debug", help="Debug mode")

    args = parser.parse_args()

    #Check the input arguments
    if not os.path.isfile(args.input):
        logging.error('---input image file does not exist---')
        sys.exit(1)

    outputpath = args.outputpath
    if outputpath != '' and not os.path.exists(outputpath):
        os.makedirs(outputpath)

    img = cv2.imread(args.input)
    (rows, cols)= (img.shape[0], img.shape[1])
    Debug = int(args.debug)

    starttime = time.time()
    text_recognition(img, Debug,0)
    #text_detect_image(img,Debug)
    endtime = time.time()

    if Debug:
        logging.debug('Text zone detection and groupping time: ' + str(endtime - starttime) +'seconds')
    sys.exit(0)



if __name__ == "__main__":

    main()
