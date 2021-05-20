from matplotlib import pyplot as plt
import os
from pycocotools.coco import COCO
from skimage import io
import time

def vis_cube(img_dir, json_path):
    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=["cube"])
    catIds = coco.loadCats(catIds)[0]['id']
    list_imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(list_imgIds[0])[0]
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)
    image = io.imread(os.path.join(img_dir, img['file_name']))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for ann in img_anns:      
        coco.showAnns([ann])

def vis_monkey(img_dir, json_path):
    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=["monkey"])
    catIds = coco.loadCats(catIds)[0]['id']
    list_imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(list_imgIds[0])[0]
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)
    image = io.imread(os.path.join(img_dir, img['file_name']))
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for ann in img_anns:      
        coco.showAnns([ann])

def main():
    img_dir = "COCO/tmp/rgb"
    json_path = "COCO/annotations/test.json"
    vis_cube(img_dir, json_path)
    vis_monkey(img_dir, json_path)
    plt.show()

if __name__ == '__main__':
    main()
