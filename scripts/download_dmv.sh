#!/bin/bash

curl -L https://github.com/OnizukaLab/nar-cardest/releases/download/v0.1.0/dmv.csv.gz | gzip -d > datasets/dmv.csv
