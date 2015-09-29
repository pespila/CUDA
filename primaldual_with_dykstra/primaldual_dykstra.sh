#!/bin/sh
#
#Script for make and run program
#
#

make

./primaldual_dykstra -i "../images/ladama.png" -o "../results/ladama/ladama_dykstra.png" -level 8 -repeats 200 -dykstra 100 -nu 0.01 -lambda 0.1 > "../results/ladama/data_dykstra.txt"
./primaldual_dykstra -i "../images/marylin.png" -o "../results/marylin/marylin_dykstra.png" -level 8 -repeats 400 -dykstra 10 -nu 0.001 -lambda 0.1 > "../results/marylin/data_dykstra.txt"
./primaldual_dykstra -i "../images/synth_gauss.png" -o "../results/synth_gauss/synth_gauss_dykstra.png" -level 8 -repeats 500 -dykstra 100 -nu 0.01 -lambda 0.11 > "../results/synth_gauss/data_dykstra.txt"
./primaldual_dykstra -i "../images/crack_tip.png" -o "../results/crack_tip/crack_tip_dykstra.png" -level 16 -repeats 400 -dykstra 100 -nu 0.01 -lambda 0.1 > "../results/crack_tip/data_dykstra.txt"
# ./primaldual_dykstra -i "../images/synth.png" -o "../results/synth/synth_dykstra.png" -level 16 -repeats 1500 -dykstra 200 -nu 0.0001 -lambda 0.1 > "../results/synth/data_dykstra.txt"

gnuplot ../results/ladama/dual_energy_dykstra.gpl
gnuplot ../results/marylin/dual_energy_dykstra.gpl
gnuplot ../results/synth_gauss/dual_energy_dykstra.gpl
gnuplot ../results/crack_tip/dual_energy_dykstra.gpl
# gnuplot ../results/synth/dual_energy_dykstra.gpl