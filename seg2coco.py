import numpy as np
import os
import cv2
import json
from pycocotools import mask
from skimage import measure
import time

def test_time(start_or_end):
    if start_or_end == 1:
        if not hasattr(test_time, 'start'):
            test_time.start = time.time()
        print("RUN: {:.3f}s".format(time.time() - test_time.start))
    else:
        test_time.start = time.time()

def gen_anno(inst_img, sem_img, image_id):
    # https://hub.fastgit.org/cocodataset/cocoapi/issues/131
    try:
        if inst_img.shape[:2] != sem_img.shape[:2]:
            raise Exception
    except:
        return None

    if len(inst_img.shape) != 2:
        inst_img = cv2.cvtColor(inst_img, cv2.COLOR_BGR2GRAY)
    if len(sem_img.shape) != 2:
        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2GRAY)
        
    height, width = inst_img.shape[:2]
    
    sub_masks = {}
    sub_masks_cls = {}
    for y in range(height):
        for x in range(width):
            pixel = inst_img[y, x]
            if pixel != 0:
                sub_mask = sub_masks.get(pixel, None)
                if sub_mask is None:
                    sub_masks[pixel] = np.zeros((height, width), dtype=np.uint8)
                    sub_masks_cls[pixel] = int(sem_img[y, x])
                sub_masks[pixel][y, x] = 255
    annotations = []
    for pixel, binary_mask in sub_masks.items():
        fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        # contours = measure.find_contours(binary_mask, 0.5) ## Highest accuracy, but takes up a lot of space
        # contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] ## Second accuracy
        # contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] ## Third accuracy
        # contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[0] ## Fourth accuracy
        contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0] ## Lowest accuracy, but save space
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
            "category_id": sub_masks_cls[pixel],
            "id": int(time.time() * 1000),
        }
        
        annotations.append(annotation)
    return annotations

def seg2coco(inst_dir, sem_dir, categories, json_filepath):
    coco_json = {
        "info": {},
        "licenses": {},
        "images": [],
        "annotations": [],
        "categories": [],
    }
    listdir = os.listdir(inst_dir)
    listdir.sort(key=lambda x: int(x.split('.')[0]))
    
    i = 0
    for filename in listdir:
        try:
            # Their file names should be same.
            inst_img = cv2.imread(os.path.join(inst_dir, filename))
            sem_img = cv2.imread(os.path.join(sem_dir, filename))
            height, width = inst_img.shape[:2]

            image_id = int(os.path.basename(filename).split('.')[0])
            image = {
                "file_name": filename.replace(".png", ".jpg"),
                "height": height,
                "width": width,
                "id": image_id,
            }

            annotation = gen_anno(inst_img, sem_img, image_id)
            if annotation is None:
                continue

            coco_json["images"].append(image)
            coco_json["annotations"].extend(annotation)

            i += 1
            if i % 100 == 0:
                print("%d is done." % i)

        except Exception as e:
            print(e)

    coco_json["categories"] = categories

    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    with open(json_filepath, "w") as f:
        f.write(json.dumps(coco_json))

def main():
    inst_dir = "COCO/tmp/inst"
    sem_dir = "COCO/tmp/sem"
    categories = [
        {"id": 1, "name": "cube", "supercategory": "object"},
        {"id": 2, "name": "monkey", "supercategory": "object"},
    ] 
    json_filepath = "COCO/annotations/test.json"

    seg2coco(inst_dir, sem_dir, categories, json_filepath)


if __name__ == '__main__':
    test_time(0)
    main()
    test_time(1)
