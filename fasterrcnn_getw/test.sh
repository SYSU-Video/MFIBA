#!/bin/bash

python3 train.py --data-path  /home/ta/liujunle/coco \
--dataset coco \
--num-classes  88 \
--batch-size 16 \
--test-only \
--model resnet50\
--pretrained True

