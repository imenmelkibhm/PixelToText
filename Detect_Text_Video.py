#This file defines the main routine for the text detection task in a given video
import logging
import traceback
import os
import numpy as np
import cv2
import argparse
import time
import sys
import operator
import math
from Detect_Text_Image import text_detect_image, text_recognition
import subprocess
from multiprocessing import Pool, cpu_count
from ChannelLogoDetection import Detect_Logo_Chanel_Frame
import ConfigParser
import glob
import pylab as pl
import shutil
import xml.etree.cElementTree as ET

def readconfig(args):
    found = False
    config = ConfigParser.ConfigParser()
    config.read('logos.conf')
    #chercher config par chaine id
    channel_id = args.id
    #chunckname=os.path.basename(args.input)
    #names=chunckname.split('_')
    #chaine=names[0].lower()

    channels_id = config.sections()
    if channel_id in channels_id:
        found = channel_id
        channel = config.get(found,"channel_name", '')
        logging.info("Config found for channel " + channel)

    if found == None:
        logging.info("Config NOT found for channel id  " + channel_id)

    else:
        if config.has_option(found, "logo_zone"):
            logo_zone=config.get(found,"logo_zone", '')
            if logo_zone != '':
                logging.info("Config of the logo zone for "+ channel + " : " + logo_zone)
                args.logo_zone = logo_zone

        if config.has_option(found, "logo_path"):
            logo_path=config.get(found,"logo_path", '')
            if logo_path != '':
                logging.info("Config of the logo path for "+ channel + " : " + logo_path)
                args.logo_path = logo_path

        if config.has_option(found, "text_zone"):
            text_zone=config.get(found,"text_zone", '')
            if text_zone != '':
                logging.info("Config of the text zone for "+ channel + " : " + text_zone)
                args.text_zone = text_zone


def removeFolders(args):

    dumpRepo = os.path.join(args.outputpath,'Dump_'+args.outputname)
    if os.path.exists(dumpRepo):
        subprocess.call('rm -rf '+ dumpRepo, shell=True)
        logging.info('Removing video dump folder!!')

    tempRepo = os.path.join(args.outputpath,'temp_'+args.outputname)
    if not args.debug:
        shutil.rmtree(tempRepo)

def dump_video(args):

    #Create the dump repo
    dumpRepo = os.path.join(args.outputpath,'Dump_'+args.outputname)
    if dumpRepo != '' and not os.path.exists(dumpRepo):
        os.makedirs(dumpRepo)
        logging.info('Create the video dump folder!!')
    #If the repo already exists
    else:
        if os.path.exists(dumpRepo):
            subprocess.call('rm -rf '+ dumpRepo, shell=True)
            os.makedirs(dumpRepo)
            logging.info('Remove old dump folder and Create the new one!!')

    subprocess.call('ffmpeg -i ' + args.input + ' -vf fps=' + args.frequency + ' ' + dumpRepo + '/frame-%d.png' + ' -threads 0', shell=True) #faster use jpg but lower image quality
    #subprocess.call('ffmpeg -i ' + args.input + ' -r ' + args.frequency + ' -f image2 ' + dumpRepo + '/frame-%d.jpg'  + ' -threads 0', shell=True)
    return dumpRepo

#Multiprocessing text detection version
def text_detect_image_parallel(arg):
    im=arg[0]
    Debug=arg[1]
    debugPath=arg[2]
    stime=arg[3]

    try:
        text_recognition(im, Debug, debugPath, stime)
        logging.info('End Text_recognition for frame {0}'.format(stime))
        return
    except OSError as e:
        s = str(e)
        logging.error('Exception: '+s)
        logging.error('OSError. stop from time frame: {0}'.format(stime))
        return  # in order to stop whole code
    except MemoryError:
        logging.error('check you available memory. stop from time frame: {0}'.format(stime))
        return
    except Exception as e:
        s = str(e)
        logging.warning('Exception: '+s)
        logging.warning('ignore time frame: {0}'.format(stime))
        return


def Detect_Logo_Chanel_parallel(arg):
    dumpRepo = arg[0]
    logo = arg[1]
    i = arg[2]
    x1 = arg[3]
    y1 = arg[4]
    x2 = arg[5]
    y2 = arg[6]

    logoIm = cv2.imread(logo)  # logo image
    if logoIm is None:
        logging.error('No logo image found for this channel!!')
        sys.exit(1)

    try:
        n_match = Detect_Logo_Chanel_Frame(dumpRepo, logoIm, i, x1, y1, x2, y2)
        return n_match, i
    except OSError:
        logging.error('OSError. stop from time frame: {0}'.format(i))
        return  # in order to stop whole code
    except MemoryError:
        logging.error('check you available memory. stop from time frame: {0}'.format(i))
        return
    except Exception as e:
        s = str(e)
        logging.warning('Exception: '+s)
        logging.warning('ignore time frame: {0}'.format(i))
        return

# create XML file from text by xml.etree.cElementTree
def xml_create(text_block, duration, stime, filename):
    hits = ET.Element("HITLIST")

    for i, text in enumerate(text_block):
        tb = (' ').join(ks for ks in text).strip()

        wordsli = re.split('[ )(":.\[\];/,\n]', tb)  # split sentence into words
        if '' in wordsli:
            wordsli.remove('')

        # save to xml structure
        hit = ET.SubElement(hits, "HIT")  # {0}".format(count))
        ET.SubElement(hit, "frame", stime="{0}".format(stime[i]),
                      dur="{0}".format(duration[i])).text = "{0} second".format(stime[i])
        for indw, word in enumerate(wordsli):
            if word == '':
                continue
            word = word.strip()
            ET.SubElement(hit, "Word", stime="{0}".format(stime[i]),
                          dur="{0}".format(duration[i])).text = "{0}".format(word)

    tree = ET.ElementTree(hits)
    tree.write(filename, xml_declaration=True, encoding='UTF-8')


#Text detection in video using multiprocessing
def text_detect_video(args):
    start = time.time()
    #Open the video and get params
    cap = cv2.VideoCapture(args.input)
    fintv = float(args.frequency)
    Debug = int(args.debug)
    fps = cap.get(5)  # frame per second in video
    frn = int(cap.get(7))  # frame number
    maxframe = int(fintv*frn/fps)

    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    debugPath=os.path.join(args.outputpath,'temp_'+args.outputname)
    if not os.path.exists(debugPath):
        os.mkdir(debugPath)

    #Multiprocessing
    pool1=Pool(processes=6)
    pool2=Pool(processes=6)
    arg_pool_logo = []
    arg_pool = []

    #1- Dump the video frames on the disk
    logging.info('(1). Dump the video frames on the disk')
    dumpRepo = dump_video(args)
    end_dump = time.time()
    logging.info('(1). Job finished in %f' % (end_dump-start))

    #2- Detect channel logo to discard unwanted frames
    logging.info('(2). Detect and discard announcement frames')
    if args.logo_zone!='' and args.logo_path!='':
        data = args.logo_zone.split(',')
        if len(data)==4 :
            x1 = int(data[0])
            y1 = int(data[1])
            x2 = int(data[2])
            y2 = int(data[3])
        path = args.logo_path
        #loop on the dumped frames
        for i in xrange(0,maxframe+1,1):
            arg_pool_logo.append([dumpRepo,path, i, x1, y1, x2, y2])

        #run multiple processing
        results = pool1.map(Detect_Logo_Chanel_parallel, arg_pool_logo)
        pool1.close()
        pool1.join()
        pool1.terminate()
        # print('Results.length = %d' %(results._length))
        # while not results:
        #     print "Waiting.."
        output= []
        for p in results:
            temp = p
            output.append(temp)
        output = np.array(output)

        end_logo = time.time()
        logging.info('(2). Job finished in %f' % (end_logo - end_dump))
    else:
        logging.error('No logo information found for this channel! Can not process this step. The whole video will be processed for text detection.')

    logging.info('(3). Text regions detection')
    #Load the corresponding mask frame for text search
    if args.text_zone!='':
        mask = cv2.imread(args.text_zone,0)
        ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    #loop on the dumped frames
    frame_pre = None
    for i in xrange(0,len(output),1):
         #if output[i, 0] > 0:
            im = cv2.imread(dumpRepo + "/frame-%d.png" %(i+1))
            if mask is not None:
                im = cv2.bitwise_and(im, im, mask = mask)
            # Check for changes with the previous frame
            flag = 0
            if frame_pre is not None:
                #compare histgram of the lower center part of the image
                f1 = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)
                f2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                hsv1 = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

                h1 = cv2.calcHist([hsv1], [0], None, [16], [0, 256])
                h2 = cv2.calcHist([hsv2], [0], None, [16], [0, 256])

                sim = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
                err = np.sum(np.sqrt((frame_pre.astype("float") - im.astype("float")) ** 2))
                err /= float(im.shape[0] * im.shape[1])

                #compare akaze features descriptor
                extractor = cv2.AKAZE_create()
                kp2, desc2 = extractor.detectAndCompute(f2, None)
                kp1, desc1 = extractor.detectAndCompute(f1, None)
                #logging.info('sim ('+str((i-int(np.round(fps / fintv))) / fps)+', '+str(i / fps)+')='+str(sim))
                if desc1 is not None and desc2 is not None and sim < 500 and np.abs(desc1.shape[0] - desc2.shape[0]) <= 30:
                    flag = 1

            frame_pre = im
            if flag == 1:
                logging.debug('skip the duplicated frame: {0} of video {1}'.format(i*int(np.round(fps/fintv)), args.input))
                continue
            logging.debug('ocr frame: {0} of video {1}'.format(i*int(np.round(fps/fintv)), args.input))
            arg_pool.append([im,Debug, debugPath, i*int(np.round(fps/fintv))])

    #run multiple processing
    pool2.imap(text_detect_image_parallel, arg_pool)
    pool2.close()
    pool2.join()
    pool2.terminate()

    #Load detected text from text files
    output=[]
    file_dir_extension=os.path.join(debugPath, 'OCR_results/*.txt')
    ocr_results_files= glob.glob(file_dir_extension)
    for file in ocr_results_files:
        stime= os.path.splitext(os.path.basename(file))[0]
        file_= open(file,'r')
        detxt= file_.read()
        output.append([stime,detxt])

    # prepare for XML
    logging.info('creating XML for video: {0}'.format(args.input))
    timestart = time.time()

    bannertext_block_all = output[:, 0]
    stime_all = output[:, 1]
    nind = np.argsort(stime_all)
    stime_all = stime_all[nind]
    bannertext_block_all = bannertext_block_all[nind]
    duration_all = np.append(stime_all[1:], np.floor(int(frn/fps))) - stime_all  # [1.0/fintv]*len(stime_all)
    # create xml file
    xml_create(bannertext_block_all, duration_all, stime_all, os.path.join(args.outputpath, os.path.basename(args.input) + '.xml'))
    timeend = time.time()
    logging.info("xml generation finised in " + str(timeend - timestart))

    logging.error('(3). Job finished in ')
    end_text = time.time()
    logging.error('(3). Job finished in %f' % (end_text-end_logo))

    #Remove the Debug and Dump folders
    removeFolders(args)

    return 0





def main():
    #Read input arguments
    parser = argparse.ArgumentParser(description='Extract the adds frames')
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-id", "--id", help="id support")
    parser.add_argument("-f", "--frequency", default=1.0, help="the frequency of extracting and processing video frames")
    parser.add_argument("-o", "--outputpath", help="the path to save output xml file")
    parser.add_argument("-n", "--outputname", help="the new file name to output")
    parser.add_argument("-lz", "--logo_zone", type=str, default='', help="zones in which we check channel logo: x1,y1,x2,y2 ")
    parser.add_argument("-l", "--logo_path", type=str, default='', help="the channel logo path")
    parser.add_argument("-tz", "--text_zone", type=str, default='', help="The text zone")
    parser.add_argument("-b", "--beginning", nargs='?', type=int, help="the adds beginning time")
    parser.add_argument("-e", "--end", nargs='?', type=int, help="the adds end time")
    parser.add_argument("-d", "--debug", nargs='?', type=int, default=0, help="Debug mode")

    args = parser.parse_args()
    readconfig(args)


    starttime = time.time()

    text_detect_video(args)


    endtime = time.time()
    logging.warning('Processing the video in %f' % (endtime-starttime))
    sys.exit(0)



if __name__ == "__main__":

    main()