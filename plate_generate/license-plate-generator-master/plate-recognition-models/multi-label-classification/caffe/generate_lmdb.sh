#!/usr/bin/env bash
python generate_txt.py

CAFFE_ROOT=/home/zhengyuwei/software/multi_label_caffe
IMAGES_DIR=/home/data_160/data3/smart_home/xiongxin-yuwei/plate_recognition/new_plate_images/
LMDB_ROOT_DIR=/home/data_160/data3/smart_home/xiongxin-yuwei/plate_recognition/new_plate_lmdb/

for MODE in all train test validate;
do
    mkdir ${LMDB_ROOT_DIR}/multi_label_lmdb
    rm -r ${LMDB_ROOT_DIR}/multi_label_lmdb/${MODE}_lmdb

    LIST_FILE=${LMDB_ROOT_DIR}/multi_label_lmdb/${MODE}.txt
    LMDB_DIR=${LMDB_ROOT_DIR}/multi_label_lmdb/${MODE}_lmdb
    SHUFFLE=true

    RESIZE_W=144
    RESIZE_H=48

    ${CAFFE_ROOT}/build/tools/convert_imageset --encoded=true --encode_type=jpg --resize_width=${RESIZE_W} --resize_height=${RESIZE_H} --shuffle=true ${IMAGES_DIR} ${LIST_FILE}  ${LMDB_DIR}
    #${CAFFE_ROOT}/build/tools/convert_imageset --encoded=true --encode_type=jpg --shuffle=true ${IMAGES_DIR} ${LIST_FILE}  ${LMDB_DIR}
done
