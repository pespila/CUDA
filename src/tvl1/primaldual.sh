#!/bin/sh
#
#Script for make and run program
#
#

make

img="../../img/"
res="./results/"

for file in "lena_noise";
do
	./primaldual -i $img$file".png" -o $res$file"/"$file".png" -repeats 1000 -tau 0.02 -lambda 1.5
done
# rm ./primaldual