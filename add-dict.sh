#!/bin/bash

for word in "$@"
do
  echo "$word" >> dict.txt
done
