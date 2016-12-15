# Exemple: python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18444715_18494715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n  iTele_20161017_18444715_18494715_SD -f 1 -d 0

from skimage.measure import structural_similarity as ssim
import logging
import os
import numpy as np
import cv2
import sys
import argparse
import numpy as np
import pylab as pl
import time
import math
import operator
import subprocess
from multiprocessing import Pool, cpu_count

#Global Variables
FLANN = 0

def Detect_Logo_SIFT_Video(args):

    Debug = int(args.debug)
    logo = cv2.imread(args.logo)  # logo image
    scale = 4
    logo = cv2.resize(logo, None, fx= scale, fy= scale, interpolation=cv2.INTER_CUBIC)

    vinput = args.input  # input video
    if not os.path.isfile(vinput):
        logging.error('---video does not exist---')
        sys.exit(1)

    cap = cv2.VideoCapture(vinput)
    logging.warning('***************************************Opening the video: ' + args.input+ ' for TV Logo detection**********************************************')
    fintv = float(args.frequency)
    fps = cap.get(5)  # frame per second in video
    frn = int(cap.get(7))  # frame number

    outputpath = args.outputpath
    if outputpath != '' and not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # verify beginning and end time
    if args.beginning is None:
        bese = 0
    else:
        bese = args.beginning
    if args.end is None:
        endse = (frn/fps)
    else:
        endse = args.end
    if bese >= endse or bese < 0 or endse > (frn/fps):
        logging.error('wrong arguments of beginning and end time')
        sys.exit(1)

    logging.info('process each segment of video {0}'.format(args.input))
    befr = int(bese * fps)  # begining frame
    endfr = int(endse * fps)  # ending frame

    n_matches = []
    frames = []
    if cap.isOpened():  # if video is opened, process frames
        ret, frame = cap.read()
        counter = 0
        #print('endfr = %d' % endfr + 'endse %d ' % endse + 'fps %d' %fps + 'frn %d' %frn )
        for i in xrange(befr, endfr, int(np.round(fps / fintv))):
            #print('i = %d' %i + '/ %d' %frn)
            while (counter != i):
                #print('counter = %d' %counter)
                ret, frame = cap.read()
                counter +=1
            #Crop the image to the ROI and zoom for better detection performances
            x1= int(args.x1)
            x2= int(args.x2)
            y1= int(args.y1)
            y2= int(args.y2)
            #print('Crop image to x1 = %d' %x1 + 'x2 = %d' %x2+'y1 = %d' %y1 + 'y2 = %d'  %y2)
            frame_ROI = frame[x1:x2,y1:y2]
            cv2.imwrite('frame_ROI.png',frame_ROI)
            scale = 4
            #frame_ROI = cv2.resize(frame_ROI, None, fx= scale, fy= scale, interpolation=cv2.INTER_CUBIC)


            #n_matches.append(Detect_Logo_SIFT_Frame(logo,frame_ROI,Debug, i))
            #n_matches.append(Detect_Logo_SURF_Frame(logo,frame_ROI,Debug, i))
            #n_matches.append(Detect_Logo_ORB_Frame(logo,frame_ROI,Debug, i))
            n_matches.append(Detect_Logo_BRISK_Frame(logo,frame_ROI,Debug, i))
            #n_matches.append(Detect_Logo_FREAK_Frame(logo,frame_ROI,Debug, i))

            frames.append(int(i/fps))

    pl.figure(figsize=(30, 4))
    chunckname_wextension=os.path.basename(args.input)
    chunckname=chunckname_wextension.split('.')[0]
    if not os.path.isfile('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt'):
        logging.warning('No ground Truth file found for commercial adds detection')
        pl.plot(frames, n_matches, 'r')
    else:
        GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt')
        GT = GT* (max(n_matches))
        print('GT dimension %d' % np.shape(GT))
        print('histo_array dimension %d' % np.shape(n_matches))
        pl.plot(frames,n_matches, 'r', label='Logo match')
        #pl.plot(frames, GT, 'g', label='Ground Truth')
        pl.legend()

    pl.savefig(os.path.join(outputpath, args.outputname + "_logoMatch.jpg"), dpi=50)
    pl.show()


def Detect_Logo_Chanel_Frame(dumpRepo, logoIm, i, x1, x2, y1, y2):
    #print "In Detect_Logo_Chanel_Frame %d " + format(i+1)
    im = cv2.imread(dumpRepo + "/frame-%d.png" %(i+1))
    if im is not None:
        frame_roi = im[x1:x2,y1:y2]
        return Detect_Logo_BRISK_Frame(logoIm,frame_roi,0, i)
    else:
        return -1

def Detect_Logo_Chanel(dumpRepo, logo, fintv, fps, frn, x1, x2, y1, y2):
    maxframe = int(fintv*frn/fps)
    logoIm = cv2.imread(logo)  # logo image
    n_matches = []
    for i in xrange(0,maxframe+1,1):
        im = cv2.imread(dumpRepo + "/frame-%d.png" %(i+1))
        if im is not None:
            frame_ROI = im[x1:x2,y1:y2]
            n_matches.append(Detect_Logo_BRISK_Frame(logoIm,frame_ROI,0, i))


    return n_matches


def Detect_Logo_SIFT_Frame(logo, frame, Debug, i):
    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initiate the SIFT descriptor
    sift = cv2.xfeatures2d.SIFT_create()

    #Find the Keypoints and descriptors with SIFT
    (kps, descs) = sift.detectAndCompute(gray, None)
    (kps2, descs2) = sift.detectAndCompute(gray_frame, None)

    dummy = np.zeros((1,1))

    #Draw the keypoints
    img=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
    img2=cv2.drawKeypoints(gray_frame,kps2,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

    if Debug:
        cv2.imwrite('sift_keypoints.jpg',img)
        cv2.imwrite('sift_keypoints2.jpg',img2)

    # Match descriptors.
    if descs2 is None:
        logging.error('No descriptor found for frame %d' % i)
        return 0

    #create a Flan matcher
    #FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descs,descs2, k=2)

    # # create BFMatcher object
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descs,descs2, k=2)

    # Apply ratio test
    good = []
    n_matches = 0
    for (m_n) in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.75*n.distance:
            good.append([m])
            n_matches+=1

    if Debug:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img_1 = cv2.drawMatchesKnn(logo, kps, frame, kps2, good, dummy, flags=2)
        pl.imshow(img_1),pl.show()

    return n_matches


def Detect_Logo_SURF_Frame(logo, frame, Debug, i):


    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initiate the SIFT descriptor
    sift = cv2.xfeatures2d.SURF_create()

    #Find the Keypoints and descriptors with SIFT
    (kps, descs) = sift.detectAndCompute(gray, None)
    (kps2, descs2) = sift.detectAndCompute(gray_frame, None)

    dummy = np.zeros((1,1))

    #Draw the keypoints
    img=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
    img2=cv2.drawKeypoints(gray_frame,kps2,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

    if Debug:
        cv2.imwrite('sift_keypoints.jpg',img)
        cv2.imwrite('sift_keypoints2.jpg',img2)


    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    if descs2 is None:
        logging.error('No descriptor found for frame %d' %i)
        return 0

    matches = bf.knnMatch(descs,descs2, k=2)

    # Apply ratio test
    good = []
    n_matches = 0
    for (m_n) in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.75*n.distance:
            good.append([m])
            n_matches+=1

    if Debug:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img_1 = cv2.drawMatchesKnn(logo, kps, frame, kps2, good, dummy, flags=2)
        pl.imshow(img_1),pl.show()

    return n_matches


def Detect_Logo_ORB_Frame(logo, frame, Debug, i):
    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initiate the SIFT descriptor
    orb = cv2.ORB_create()

    #Find the Keypoints and descriptors with SIFT
    (kps, descs) = orb.detectAndCompute(gray, None)
    (kps2, descs2) = orb.detectAndCompute(gray_frame, None)

    dummy = np.zeros((1,1))

    #Draw the keypoints
    img=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
    img2=cv2.drawKeypoints(gray_frame,kps2,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

    if Debug:
        cv2.imwrite('sift_keypoints.jpg',img)
        cv2.imwrite('sift_keypoints2.jpg',img2)

    # Match descriptors.
    if descs2 is None:
        logging.error('No descriptor found for frame %d' % i)
        return 0

    global FLANN
    if FLANN:
        #create a Flan matcher
        #FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descs,descs2, k=2)

    else:
        # create BFMatcher object
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs,descs2, k=2)

    # Apply ratio test
    good = []
    n_matches = 0
    for (m_n) in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.75*n.distance:
            good.append([m])
            n_matches+=1

    if Debug:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img_1 = cv2.drawMatchesKnn(logo, kps, frame, kps2, good, dummy, flags=2)
        pl.imshow(img_1),pl.show()

    return n_matches


def Detect_Logo_BRISK_Frame(logo, frame, Debug, i):
    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initiate the SIFT descriptor
    brisk = cv2.BRISK_create()

    #Find the Keypoints and descriptors with SIFT
    (kps, descs) = brisk.detectAndCompute(gray, None)
    (kps2, descs2) = brisk.detectAndCompute(gray_frame, None)

    dummy = np.zeros((1,1))

    if Debug:
    #Draw the keypoints
        img=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
        img2=cv2.drawKeypoints(gray_frame,kps2,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
        cv2.imwrite('sift_keypoints.jpg',img)
        cv2.imwrite('sift_keypoints2.jpg',img2)

    # Match descriptors.
    if descs2 is None:
        #logging.error('No descriptor found for frame %d' % i)
        return 0

    global FLANN
    if FLANN:
        #create a Flan matcher
        #FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descs,descs2, k=2)

    else:
        # # create BFMatcher object
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs,descs2, k=2)

    # Apply ratio test
    good = []
    n_matches = 0
    for (m_n) in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.75*n.distance:
            good.append([m])
            n_matches+=1

    if Debug:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img_1 = cv2.drawMatchesKnn(logo, kps, frame, kps2, good, dummy, flags=2)
        pl.imshow(img_1),pl.show()

    return n_matches

def Detect_Logo_FREAK_Frame(logo, frame, Debug, i):
    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initiate the FREAK descriptor
    freak = cv2.xfeatures2d.FREAK_create()
    brisk = cv2.xfeatures2d.SURF_create()


    #Find the Keypoints  with BRISK
    kps = brisk.detect(gray, None)
    kps2= brisk.detect(gray_frame, None)

    #Find the descriptor  with FREAK
    kps, descs = freak.compute(gray, kps)
    kps2, descs2 = freak.compute(gray_frame, kps2)

    dummy = np.zeros((1,1))

    #Draw the keypoints
    img=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
    img2=cv2.drawKeypoints(gray_frame,kps2,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

    if Debug:
        cv2.imwrite('sift_keypoints.jpg',img)
        cv2.imwrite('sift_keypoints2.jpg',img2)

    # Match descriptors.
    if descs2 is None:
        logging.error('No descriptor found for frame %d' % i)
        return 0

    global FLANN
    if FLANN:
        #create a Flan matcher
        #FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descs,descs2, k=2)

    else:
        # # create BFMatcher object
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs,descs2, k=2)

    # Apply ratio test
    good = []
    n_matches = 0
    for (m_n) in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.75*n.distance:
            good.append([m])
            n_matches+=1

    if Debug:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img_1 = cv2.drawMatchesKnn(logo, kps, frame, kps2, good, dummy, flags=2)
        pl.imshow(img_1),pl.show()

    return n_matches


def main():
    parser = argparse.ArgumentParser(description='Detect channel logo')
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-l", "--logo", help="TV logo file")
    parser.add_argument("-x1", "--x1", help="x1")
    parser.add_argument("-x2", "--x2", help="x2")
    parser.add_argument("-y1", "--y1", help="y1")
    parser.add_argument("-y2", "--y2", help="y2")
    parser.add_argument("-o", "--outputpath", help="the path to save the output files")
    parser.add_argument("-n", "--outputname", help="the name to save the adds frames")
    parser.add_argument("-f", "--frequency", default=1.0, help="the frequency of extracting and processing video frames")
    parser.add_argument("-b", "--beginning", nargs='?', type=int, help="the optional beginning time")
    parser.add_argument("-e", "--end", nargs='?', type=int, help="the optional end time")
    parser.add_argument("-d", "--debug", help="Debug mode")

    args = parser.parse_args()

    starttime = time.time()

    Detect_Logo_SIFT_Video(args)


    endtime = time.time()
    logging.warning('Processing the video in %f' % (endtime-starttime))
    sys.exit(0)


if __name__ == "__main__":

    main()