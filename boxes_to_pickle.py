#!/usr/bin/env python3
"""Convert person detections from the patched Darknet output to a pickle format.

The patched Darknet output is like:

Enter Image Path: /some/path1.jpg: Predicted in 0.035027 seconds.
cell phone: 12.924%
Box (LTWH): 1319,367,75,120
car: 86.035%
truck: 13.739%
Box (LTWH): 1799,345,79,47
Enter Image Path: /some/path2.jpg: Predicted in 0.035093 seconds.
cell phone: 14.358%
Box (LTWH): 1320,367,1333,382

So per image it outputs multiple boxes, and for each box multiple class labels, each with its own
confidence.
"""

import argparse
import logging
import os.path
import re
import sys

import numpy as np


def main():
    flags = initialize()
    logging.info(f'Opening {flags.in_path}')
    with open(flags.in_path, 'r') as f:
        darknet_output_text = f.read()

    detections_per_image = {}

    if flags.relpath_components == 'auto':
        matches = re.finditer(IMAGE_REGEX, darknet_output_text, flags=re.MULTILINE | re.DOTALL)
        paths = [m['path'] for m in matches]
        prefix = os.path.commonprefix(paths)
        n_components = os.path.relpath(paths[0], prefix).split('/')
        if not all(os.path.relpath(p, prefix).split('/') == n_components for p in paths):
            raise Exception

        flags.relpath_components = n_components
    else:
        flags.relpath_components = int(flags.relpath_components)

    for m_image in re.finditer(IMAGE_REGEX, darknet_output_text, flags=re.MULTILINE | re.DOTALL):
        relative_image_path = last_path_components(m_image['path'], flags.relpath_components)
        detections_per_image[relative_image_path] = []

        for m_object in re.finditer(OBJECT_REGEX, m_image['objects'], flags=re.MULTILINE):
            if m_object is None:
                continue

            bbox = m_object_to_bbox(m_object)
            if not is_shape_plausible(bbox):
                continue

            for m_class in re.finditer(CLASS_REGEX, m_object['classes'], flags=re.MULTILINE):
                if m_class['classname'] == 'person':
                    confidence = float(m_class['conf']) / 100
                    bbox_with_confidence = [*bbox, confidence]
                    detections_per_image[relative_image_path].append(bbox_with_confidence)

        if not detections_per_image[relative_image_path]:
            logging.warning(f'No detections in {relative_image_path}')

    logging.info(f'Number of images: {len(detections_per_image)}')
    n_detections = sum(len(v) for v in detections_per_image.values())
    logging.info(f'Total number of detections: {n_detections}')
    logging.info(f'Saving file to {flags.out_path}')

    with open(flags.out_path, 'wb') as f:
        pickle.dump(detections_per_image, f, protocol=pickle.HIGHEST_PROTOCOL)


def m_object_to_bbox(m_object):
    x1, y1, w, h = [int(x) for x in re.findall(r'\d+', m_object['coords'])]
    return np.array([x1, y1, w, h])


def is_shape_plausible(bbox):
    x, y, w, h = bbox
    aspect_ratio = w / h
    return w > 30 and 1 / 15 < aspect_ratio < 15


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='error')
    parser.add_argument('--relpath-components', default='auto')
    flags = parser.parse_args()
    if flags.out_path is None:
        flags.out_path = flags.in_path.replace('.txt', '.pickle')

    loglevel = dict(error=30, warning=20, info=10)[flags.loglevel]
    simple_formatter = logging.Formatter('{asctime}-{levelname:^1.1} -- {message}', style='{')
    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setLevel(loglevel)
    print_handler.setFormatter(simple_formatter)
    logging.basicConfig(level=loglevel, handlers=[print_handler])
    return flags


def split_path(path):
    return os.path.normpath(path).split(os.path.sep)


def last_path_components(path, n_components):
    components = split_path(path)
    return os.path.sep.join(components[-n_components:])


IMAGE_REGEX = r"""Enter Image Path: (?P<path>.+?): Predicted in .+? seconds\.
(?P<objects>.*?)(?=Enter)"""
OBJECT_REGEX = r"""(?P<classes>(?:.+?: .+?%
)+)Box \(LTWH\): (?P<coords>.+?)
"""
CLASS_REGEX = r"""(?P<classname>.+?): (?P<conf>.+)%"""

if __name__ == '__main__':
    main()
