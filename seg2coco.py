import numpy as np
import pandas as pd
import os
import cv2
import json
from matplotlib import pyplot as plt
import scipy.io as scio
from PIL import Image  
from pycocotools import mask
from pycocotools.coco import COCO
from skimage import measure, io
import time

def test_time(start_or_end):
    if start_or_end == 1:
        if not hasattr(test_time, 'start'):
            test_time.start = time.time()
        print("RUN: {:.3f}s".format(time.time() - test_time.start))
    else:
        test_time.start = time.time()

def gen_anno(seg_img, image_id):
    if len(seg_img.shape) != 2:
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    height, width = seg_img.shape[:2]
    
    sub_masks = {}
    for y in range(height):
        for x in range(width):
            pixel = seg_img[y, x]
            if pixel != 0:
                sub_mask = sub_masks.get(pixel, None)
                if sub_mask is None:
                    sub_masks[pixel] = np.zeros((height, width), dtype=np.uint8)
                sub_masks[pixel][y, x] = 255
    annotations = []
    for binary_mask in sub_masks.values():
        fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(binary_mask, 0.5)
        segmentations = []

        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentations.append(segmentation)

        annotation = {
            "segmentation": segmentations,
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": int(time.time() * 1000),
        }
        
        annotations.append(annotation)
    return annotations

def main():
    coco_json = {
        "info": {},
        "licenses": {},
        "images": [],
        "annotations": [],
        "categories": [],
    }
    seg_dir = "COCO/tmp/inst"
    listdir = os.listdir(seg_dir)
    listdir.sort(key=lambda x: int(x.split('.')[0]))

    i = 0
    for filename in listdir:
        filepath = os.path.join(seg_dir, filename)
        # print(filepath)
        try:
            seg_img = cv2.imread(filepath)
            height, width = seg_img.shape[:2]

            image_id = int(os.path.basename(filename).split('.')[0])
            image = {
                "file_name": filename.replace(".png", ".jpg"),
                "height": height,
                "width": width,
                "id": image_id,
            }

            annotation = gen_anno(seg_img, image_id)
            coco_json["images"].append(image)
            coco_json["annotations"].extend(annotation)

            i += 1
            if i % 100 == 0:
                print("%d is done." % i)

        except Exception as e:
            print(e)

    coco_json["categories"] = [
        {"id": 1, "name": "target", "supercategory": "target"}
    ] 
    with open("COCO/annotations/{}.json".format(int(time.time() * 1000)), "w") as f:
        f.write(json.dumps(coco_json))

    

if __name__ == '__main__':
    test_time(0)
    main()
    test_time(1)
