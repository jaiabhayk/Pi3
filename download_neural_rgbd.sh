#!/bin/bash
mkdir ./data/neural_rgbd
cd ./data/neural_rgbd
wget http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
unzip neural_rgbd_data.zip
rm neural_rgbd_data.zip
cd -
