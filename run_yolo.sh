#!/usr/bin/env bash
shopt -s extglob

IMG_ROOT=$1
DESTINATION=$2
NJOBS=$3
DARKNET_DIR=.

if [[ -d $DESTINATION ]]; then
    echo "$DESTINATION must not be a directory, it should be a target file name"
    exit 1
fi

find "$IMG_ROOT" -name '*.jpg' \
 | sort
 | parallel --jobs $NJOBS -N1 --round-robin --roundrobin --pipe \
    "$DARKNET_DIR/darknet" detect "$DARKNET_DIR/cfg/yolov3-spp.cfg" "$DARKNET_DIR/yolov3-spp.weights" -thresh 0.1 \
    '>>' "${DESTINATION}{%}.txt"

cat "${DESTINATION}*.txt" > "${DESTINATION}_all.txt"
rm "${DESTINATION}!(_all).txt"
$DARKNET_DIR/boxes_to_pickle.py --in-path "${DESTINATION}_all.txt"
