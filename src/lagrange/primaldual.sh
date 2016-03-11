#!/bin/sh
#
#Script for make and run program
#
#

make

for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do
	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
	do
		echo $i" "$j
		./primaldual -i "../../img/ladama.png" -o "../../tests/parameter/lagrange/ladama/"$i"ladama"$j".png" -repeats 10000 -lambda $i -nu $j -gray -level 16 >> "../../tests/parameter/lagrange/ladama/output.txt"
		./primaldual -i "../../img/synth.png" -o "../../tests/parameter/lagrange/synth/"$i"synth"$j".png" -repeats 10000 -lambda $i -nu $j -gray -level 16 >> "../../tests/parameter/lagrange/synth/output.txt"
		./primaldual -i "../../img/synth_gauss.png" -o "../../tests/parameter/lagrange/synth_gauss/"$i"synth_gauss"$j".png" -repeats 10000 -lambda $i -nu $j -gray -level 16 >> "../../tests/parameter/lagrange/synth_gauss/output.txt"
		./primaldual -i "../../img/crack_tip.png" -o "../../tests/parameter/lagrange/crack_tip/"$i"crack_tip"$j".png" -repeats 10000 -lambda $i -nu $j -gray -level 16 >> "../../tests/parameter/lagrange/crack_tip/output.txt"
		./primaldual -i "../../img/hepburn.png" -o "../../tests/parameter/lagrange/hepburn/"$i"hepburn"$j".png" -repeats 10000 -lambda $i -nu $j -gray -level 16 >> "../../tests/parameter/lagrange/hepburn/output.txt"
	done
done