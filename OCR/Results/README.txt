A C++/Python program for text detection and recognition in video stream.
This folder contains the following files:
    - Detect_Text_Video.py
    - Detect_Text_Image.py
    - ChannelLogoDetection.py
    - trained_classifierNM1.xml
    - trained_classifierNM2.xml
    - trained_classifier_erGrouping.xml
    - TextMSER/text_recongnition.cpp


#Installing FFmpeg 2.8 from jessie-backports repository:

a. add 'deb http://http.debian.net/debian jessie-backports main' into /etc/apt/sources.list
b. sudo apt-get update
c. sudo apt-get -t jessie-backports install ffmpeg





#Compile source
python ./Detect_Text_Image.py -i /opt/exe/opencv_contrib/modules/text/samples/scenetext01.jpg -o Demo -d 1

g++ -shared -Wl,-soname,TextDetect -o Text.so -ggdb `pkg-config --cflags --libs opencv` -fPIC text_recongnition.cpp


g++ -shared -Wl,-soname,TextDetect -o Text.so -ggdb `pkg-config --cflags --libs opencv` -lboost_system -lboost_filesystem -fPIC text_recongnition.cpp