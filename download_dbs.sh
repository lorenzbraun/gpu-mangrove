#!/usr/bin/env bash

if [ ! -f data.tar.gz ]; then
  echo downloading data.tar.gz
  wget -O data.tar.gz https://heibox.uni-heidelberg.de/f/938997f66aef46fc9188/?dl=1
fi

if [ ! -d data ]; then
  echo extracting data.tar.gz
  tar xzf data.tar.gz
fi
