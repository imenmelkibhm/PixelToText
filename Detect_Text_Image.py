#This file includes the core function for text detection in a given image

import sys
import os
import argparse
import numpy as np
import re
import cv2
import time
import logging
from ctypes import *
from PIL import Image, ImageEnhance
from itertools import takewhile
import xml.etree.cElementTree as ET

#Define logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# create XML file from text by xml.etree.cElementTree
def xml_create_unit(text_block, duration, stime, keywords, filename):
    hits = ET.Element("HITLIST")

    for i, text in enumerate(text_block):

        tb = (' ').join(ks for ks in text).strip()

        wordsli = re.split('[ )(":.\[\];/,\n]', tb) # split sentence into words
        if '' in wordsli:
            wordsli.remove('')
        begrec_all=[]
        wordrec_all=[]
        temprec_all=[]
        lenrec_all=[]
        # match with keywords from file
        for wordnum in xrange(len(wordsli),0,-1):
            for indword in xrange(len(wordsli)-wordnum+1): # check multiple words unit in a line
                word =u''
                for s in xrange(wordnum):
                    word += (u' ' + wordsli[indword+s])
                word_without_acc = unicodedata.normalize('NFKD', word.strip()).encode('ASCII','ignore') # convert characters with French accent to normal English chracters
                word=word.strip().encode('utf8')
                flag2=0
                # French accent matching
                if (word.lower() in keywords)>0:#np.sum([find_substring(word.lower(), s) for s in keywords]) > 0:# French words matching
                    pla = np.where(keywords==word.lower())[0]
                    temprec_all = temprec_all + pla.tolist()
                    begrec_all = begrec_all+ [indword]*len(pla)
                    wordrec_all = wordrec_all+ [word]*len(pla)
                    lenrec_all = lenrec_all + [wordnum]*len(pla)
                    flag2=1
                # matching without accent
                if flag2==0 and (word_without_acc.lower() in keywords)>0:#np.sum([find_substring(word.lower(), s) for s in keywords]) > 0:# converted English words matching
                    pla = np.where(keywords==word_without_acc.lower())[0]
                    temprec_all = temprec_all + pla.tolist()
                    begrec_all = begrec_all+ [indword]*len(pla)
                    wordrec_all = wordrec_all+ [word]*len(pla)
                    lenrec_all = lenrec_all + [wordnum]*len(pla)

        # save to xml structure
        if len(temprec_all)==0:#if no matched keywords with Keywords list
            hit = ET.SubElement(hits,"HIT")# {0}".format(count))
            ET.SubElement(hit, "frame", stime="{0}".format(stime[i]), dur="{0}".format(duration[i])).text = "{0} second".format(stime[i])
            for indw, word in enumerate(wordsli):
                if word=='':
                        continue
                word=word.strip()
                ET.SubElement(hit, "Word", stime="{0}".format(stime[i]), dur="{0}".format(duration[i])).text = "{0}".format(word)
        else: # if there is matched words to keywords
            count=1
            while len(temprec_all) > 0:# multiple matched words or word segments
                flag = 1
                wordrec = wordrec_all.pop()
                temprec = temprec_all.pop()
                begrec = begrec_all.pop()
                lenrec = lenrec_all.pop()

                hit = ET.SubElement(hits,"HIT", DUP="{0}".format(count))
                ET.SubElement(hit, "frame", stime="{0}".format(stime[i]), dur="{0}".format(duration[i])).text = "{0} second".format(stime[i])
                for indw, word in enumerate(wordsli):# if there is multiple mathced keywords in a frame, output for multiple times
                    if word=='':
                        continue
                    word=word.strip()

                    if indw==begrec and flag == 1:#np.sum([find_substring(word.lower(), s) for s in keywords]) > 0:#
                        ET.SubElement(hit, "Word", keyword="{0}".format(wordrec),  classement="{0}".format(keywords[temprec,1]), nature="{0}".format(keywords[temprec,0]), stime="{0}".format(stime[i]), dur="{0}".format(duration[i])).text = "{0}".format(wordrec)
                        flag=2
                    elif flag==2 and begrec+lenrec>indw>begrec :
                        pass
                    else:
                        ET.SubElement(hit, "Word", stime="{0}".format(stime[i]), dur="{0}".format(duration[i])).text = "{0}".format(word)

                count +=1
    tree = ET.ElementTree(hits)
    tree.write(filename, xml_declaration=True, encoding='UTF-8')


def text_recognition(img, Debug, debugPath,i):
    #Preprocessing to enhance text structure in the image
    #Image enhancement using PIL
    image = Image.fromarray(img)
    contrast = ImageEnhance.Contrast(image)
    contrasted = contrast.enhance(1)

    #contrasted.show()
    img = np.array(contrasted)
    (rows, cols) = (img.shape[0], img.shape[1])

    #sharpen the image
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    img = cv2.filter2D(img, -1, kernel)

    # Load the dll
    mydll = cdll.LoadLibrary("TextMSER/libText.so")
    #mydll.text_recognition.restype = c_char_p
    mydll.text_recognition(img.ctypes.data_as(POINTER(c_ubyte)), rows, cols,i, Debug, debugPath, 0)


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
        er1 = cv2.text.createERFilterNM1(erc1, 8, 0.00015, 0.13, 0.2, True, 0.1)#16

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
