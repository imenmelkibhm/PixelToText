
import ConfigParser
import os
import logging
import argparse
import sys
import cv2
import subprocess
from multiprocessing import Pool, cpu_count
import time



def readconfig(args):
    found = False
    config = ConfigParser.ConfigParser()
    config.read('logos.conf')
    #chercher config par chaine
    chunckname=os.path.basename(args.input)
    names=chunckname.split('_')
    chaine=names[0].lower()

    logging.info("config trouvee pour la chaine " + chaine)
    chaines = config.sections()
    if chaine in chaines:
        found = chaine
        logging.info("config trouvee pour la chaine " + chaine)

    else:
        for c in chaines:
            if chaine.startswith(c) :
                found = c
                logging.info("config  trouvee pour la chaine " + chaine)
                break

    if found == None:
        logging.info("config non trouvee pour la chaine " + chaine)

    else:
        if config.has_option(found, "logo"):
            logo=config.get(found,"logo", '')
            if logo != '':
                logging.info("config de logo pour "+found + " : " + logo)
                args.logo = logo


def dump_video(args):

    #Create the dump repo
    dumpRepo = args.dumprepo
    if dumpRepo != '' and not os.path.exists(dumpRepo):
        os.makedirs(dumpRepo)
        print 'Create the dump folder!!'
    #If the depo already exists
    else:
        if os.path.exists(dumpRepo):
            subprocess.call('rm -rf '+ dumpRepo, shell=True)
            print 'removing old folder!!'
            os.makedirs(dumpRepo)
            print 'And create the dump folder!!'

    subprocess.call('ffmpeg -i ' + args.input + ' -vf fps=' + args.frequency + ' ' + dumpRepo + '/frame-%d.png' + ' -threads 0', shell=True) #faster use jpg but lower image quality


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - line %(lineno)d - %(process)d - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='parallel OCR of video.')
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-f", "--frequency", default=1.0, help="the frequency of extracting and processing video frames")
    parser.add_argument("-dr", "--dumprepo", help="the temporary folder used to dump the video images")
    parser.add_argument("-l", "--logo", type=str, default='', help="zones in which we check chain logo: x1,y1,x2,y2 ")


    args = parser.parse_args()
    readconfig(args)
    dumpRepo = args.dumprepo
    #1- Dump the video frames on the disk
    logging.info('(1). Dump the video frames on the disk')
    dump_video(args)

    if args.logo != '' :
        data = args.logo.split(',')
        if len(data)==4 :
            x1 = int(data[0])
            y1 = int(data[1])
            x2 = int(data[2])
            y2 = int(data[3])
    for i in xrange(0,301,1):
        image = cv2.imread(dumpRepo + "/frame-%d.png" %(i+1))
        image_croped= image[y1:y2,x1:x2]
        cv2.imshow("image cropped", image_croped)

    cv2.waitKey(0)
    sys.exit(0)

if __name__ == "__main__":

    #cProfile.run('main()')
    main()