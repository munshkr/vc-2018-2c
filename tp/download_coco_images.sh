#!/bin/bash
for file in test2017 val2017; do
  wget --continue http://images.cocodataset.org/zips/${file}.zip
  unzip $file
done
