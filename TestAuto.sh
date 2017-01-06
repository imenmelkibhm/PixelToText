#!/usr/bin/env bash

rm -f Debug_*

dir=/opt/exe/textocr/demo/Chunks/TF1
cd $dir
chunk_list=`ls | grep _SD.ts`

cd /opt/exe/PixelToText
for chunk in $chunk_list
do
    echo Text Detection for $chunk
    name=`basename $chunk .ts`
    python Detect_Text_Video.py -i $dir/$chunk -f 1 -dr $name -d 1
    mv Debug_images Debug_$name
done


dir=/opt/exe/textocr/demo/Chunks/BFMTV
cd $dir
chunk_list=`ls | grep _SD.ts`

cd /opt/exe/PixelToText
for chunk in $chunk_list
do
    echo Text Detection for $chunk
    name=`basename $chunk .ts`
    python Detect_Text_Video.py -i $dir/$chunk -f 1 -dr $name -d 1
    mv Debug_images Debug_$name
done


dir=/opt/exe/textocr/demo/Chunks/iTele
cd $dir
chunk_list=`ls | grep _SD.ts`

cd /opt/exe/PixelToText
for chunk in $chunk_list
do
    echo Text Detection for $chunk
    name=`basename $chunk .ts`
    python Detect_Text_Video.py -i $dir/$chunk -f 1 -dr $name -d 1
    mv Debug_images Debug_$name
done

exit


dir=/opt/exe/textocr/demo/Chunks/LCI
cd $dir
chunk_list=`ls | grep _SD.ts`

cd /opt/exe/PixelToText
for chunk in $chunk_list
do
    echo Text Detection for $chunk
    name=`basename $chunk .ts`
    python Detect_Text_Video.py -i $dir/$chunk -f 1 -dr $name -d 1
    mv Debug_images Debug_$name
done

dir=/opt/exe/textocr/demo/Chunks/France3
cd $dir
chunk_list=`ls | grep _SD.ts`

cd /opt/exe/PixelToText
for chunk in $chunk_list
do
    echo Text Detection for $chunk
    name=`basename $chunk .ts`
    python Detect_Text_Video.py -i $dir/$chunk -f 1 -dr $name -d 1
    mv Debug_images Debug_$name
done









python Detect_Text_Video.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18444715_18494715_SD.ts -f 1 -dr Dump_iTele_20161017_18444715_18494715 -d 1
mv Debug_images Debug_iTele_20161017_18444715_18494715_SD

python Detect_Text_Video.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18444715_18494715_SD.ts -f 1 -dr Dump_iTele_20161017_18444715_18494715 -d 1
mv Debug_images Debug_iTele_20161017_18444715_18494715_SD

exit
