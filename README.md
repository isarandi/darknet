This is a small fork of Darknet.

Darknet is an open source neural network framework written in C and CUDA by Joseph Redmon. For more info see http://pjreddie.com/darknet and https://github.com/pjreddie/darknet.

This fork changes the following:

* No images with bounding boxes are generated
* The bounding box coordinates are written to standard output
* The confidence is printed as a float with 3 decimals, not as an integer.
* A run_yolo.sh script to make it easier to generate boxes for all images within a directory tree.
* A script called boxes_to_pickle.py that parses the standard output of this patched darknet and collects person detections in a Python dictionary and saves it in Pickle format.

Usage:

```
git clone https://github.com/isarandi/darknet.git
cd darknet
wget https://pjreddie.com/media/files/yolov3-spp.weights
./run_yolo.sh --image-root some/path/with/images --out-path some/target/path.pkl --jobs 3
```

Fork by István Sárándi, RWTH Aachen University.
