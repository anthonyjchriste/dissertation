#!/bin/bash

for dot_file in *.dot 
do
	png_file="${dot_file%.dot}.png"
	dot -Tpng ${dot_file} > ${png_file}
	echo "Converted ${dot_file} => ${png_file}"
done
