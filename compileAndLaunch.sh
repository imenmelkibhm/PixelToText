#!/bin/bash

echo   -----------------------Compiling--------------------- 
cd TextMSER
g++ -shared -Wl,-soname,TextDetect -o Text.so -ggdb `pkg-config --cflags --libs opencv` -lboost_system -lboost_filesystem -fPIC text_recongnition.cpp

echo ----------------------Executing the python script---------------------
cd ..
python Detect_Text_Video.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18444715_18494715_SD.ts -f 1 -dr Dump_iTele_20161017_18444715_18494715 -d 1

exit 0
