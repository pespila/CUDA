#!/bin/sh
#
#Script for make and run program
#
#

make

img="../../img/"
res="./results/"

for file in "lena_noisy";
do
	./primaldual -i $img$file".png" -o $res$file"/"$file".png" -repeats 20 -tau 0.0025 -lambda 1
done
rm ./primaldual