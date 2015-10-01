#!/bin/sh
#
#Script for make and run program
#
#

make

img="../../img/"
res="./results/"

for file in "hepburnc";
do
	./primaldual -i $img$file".png" -o $res$file"/"$file".png" -repeats 100 -alpha 10 -lambda 0.1 -tau 0.25 -sigma 0.5
done
# rm ./primaldual