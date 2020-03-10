#!/usr/bin/env bash
set -euo pipefail

while [[ "$#" -gt 0 ]]; do case $1 in
  --image-root=*) IMG_ROOT="${1#*=}"; shift;;
  --out-path=*) DESTINATION="${1#*=}"; shift;;
  --jobs=*) NJOBS="${1#*=}"; shift;;
  --image-root|--out-path|--jobs) echo "$1 requires an argument" >&2; exit 1;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

if [[ -d $DESTINATION ]]; then
    echo "$DESTINATION must not be a directory, it should be a target file name"
    exit 1
fi

find "$IMG_ROOT" -name '*.jpg' | sort \
 | parallel --verbose --jobs "$NJOBS" -N1 --round-robin --pipe \
    ./darknet detect ./cfg/yolov3-spp.cfg ./yolov3-spp.weights -thresh 0.1 '>>' "${DESTINATION}.raw_{%}"

cat "${DESTINATION}.raw_*" > "${DESTINATION}.raw"
rm "${DESTINATION}.raw_*"
"$DARKNET_DIR/boxes_to_pickle.py" --in-path "$DESTINATION.raw" --out-path="$DESTINATION" --root-dir="$IMG_ROOT"
