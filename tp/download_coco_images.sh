#!/bin/bash
file=images_coco_test2017.zip
wget --continue http://images.cocodataset.org/zips/test2017.zip -O $file
unzip $file
