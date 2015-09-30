#!/bin/sh
#
#Script for make and run program
#
#

make

# -level 8 -repeats 200 -dykstra 100 -nu 0.01 -lambda 0.1 > "../results/ladama/data_dykstra.txt"
# -level 8 -repeats 400 -dykstra 10 -nu 0.001 -lambda 0.1 > "../results/marylin/data_dykstra.txt"
# -level 8 -repeats 500 -dykstra 100 -nu 0.01 -lambda 0.11 > "../results/synth_gauss/data_dykstra.txt"
# -level 16 -repeats 400 -dykstra 100 -nu 0.01 -lambda 0.1 > "../results/crack_tip/data_dykstra.txt"
# -level 16 -repeats 1500 -dykstra 200 -nu 0.0001 -lambda 0.1 > "../results/synth/data_dykstra.txt"

file="synth"
nrj="data.txt"
img="../../img/"
res="./results/"
par=$res$file"/parameter.txt"
out=$res$file"/dual_energy.png"
./primaldual -i $img$file".png" -o $res$file"/"$file".png" -data $nrj -parm $par -level 8 -repeats 400 -dykstra 100 -nu 0.001 -lambda 0.11
gnuplot -e "outfile='"$out"'" -e "datafile='data.txt'" plot.gpl
rm data.txt
rm ./primaldual

# for file in "synth" "lena" "hepburn" "ladama" "marylin" "synth_gauss" "crack_tip" "inpaint";
# do
# 	.primaldual -i $img$file".png" -o $res$file"/"$file".png" -data $nrj -parm $par -level 8 -repeats 1000 -nu 0.001 -lambda 0.1
# done