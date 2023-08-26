import json
import os
import torch
import torch.backends.cudnn as cudnn
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import get_config
from data.coco import COCODetection
from detector import Detector

import cv2 

def test_coco(model, testset, dataset_name, filename=None):
    """ Evaluates model on given dataset
    Args:
        model (Detector): detection model
        testset (COCODetection): COCO dataset
        dataset_name: name of test dataset
        filename (str): if not None, file where to write results
    Returns:
        list: list of dictionaries with:
            'image_id' (int): image id in dataset
            'category_id (int): predicted category id
            'bbox' (list): predicted (x, y, w, h)
            'score' (float): predicted probability
    """
    num_images = len(testset)
    result = []
    tq = tqdm.tqdm(total=num_images)
    tq.set_description("Processing images from {}".format(dataset_name))
    for i in range(num_images):
        img = testset.pull_image(i)
        bboxes, labels, scores = model(img)
        for bbox, label, score in zip(bboxes, labels, scores):
            # added in by myself to control threshold
            if score > 0.5:
                category = {1: "tiger"}
                cv2.rectangle(
                            img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),
                            color=(0,255,0), thickness=2
                        )
                img = cv2.putText(img, category[label] + " - " + str(f"{score:.2f}"), (int(bbox[0]), int(bbox[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,0), thickness=2, 
                    lineType=cv2.LINE_AA)
                result.append({
                    "image_id": testset.ids[i],
                    "category_id": label,
                    "bbox": [round(i) for i in bbox],
                    "score": score
                })
            save_name = "0"* (4-len(str(testset.ids[i]))) + str(testset.ids[i])
            cv2.imwrite(f"eval/images/{save_name}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tq.update()
    tq.close()

    if filename:
        with open(filename, "w") as f:
            json.dump(result, f)
    return result

def print_summary(ann_path, result_path, dataset_name):
    """ Prints summary of a given submission file in COCO forma
    Args:
        ann_path (str): path to json file with annotations
        result_path (str): path to file containing results in COCO format
        dataset_name (str): dataset name
    Returns:
        None
    """
    cocoGt = COCO(ann_path)
    cocoDt = cocoGt.loadRes(result_path)

    imgs = cocoGt.getImgIds()
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgs
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("{0} {1} {0}".format("=" * 30, dataset_name))
    cocoEval.summarize()


if __name__ == '__main__':
    config = get_config("config/test.yaml")

    cudnn.benchmark = True
    detector = Detector(config["model"])

    if not os.path.exists(config["save_folder"]):
        os.mkdir(config["save_folder"])

    for data_config in config["data"]:
        testset = COCODetection(data_config["ann_path"], data_config["img_path"])

        save_filename = os.path.join(config["save_folder"], "{}.txt".format(data_config["name"]))

        test_coco(detector, testset, data_config["name"], save_filename)
        print_summary(data_config["ann_path"], save_filename, data_config["name"])