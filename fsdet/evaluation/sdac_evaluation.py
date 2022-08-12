import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import logging

from functools import lru_cache
from collections import OrderedDict, defaultdict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm

from fsdet.evaluation.evaluator import DatasetEvaluator

class SDACEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name): # initial needed variables
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._base_annotations_directory = os.path.join(meta.dirname, "annots")
        self._base_images_directory = os.path.join(meta.dirname, "images")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        self._base_classes = meta.base_classes
        self._novel_classes = meta.novel_classes
        self._cpu_device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")  # TODO gpu doesn't work yet
        self._logger = logging.getLogger(__name__)

    def reset(self): # reset predictions
        self._predictions = defaultdict(
            list
        )

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(f"Evaluating {self._dataset_name}...")

        with tempfile.TemporaryDirectory(prefix="sdac_eval") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            exist_base, exist_novel = False, False
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = sdac_eval(
                        res_file_template,
                        self._base_annotations_directory,
                        self._base_images_directory,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )
                    aps[thresh].append(ap * 100)

                    if (
                        self._base_classes is not None
                        and cls_name in self._base_classes
                    ):
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if (
                        self._novel_classes is not None
                        and cls_name in self._novel_classes
                    ):
                        aps_novel[thresh].append(ap * 100)
                        exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {
            "AP": np.mean(list(mAP.values())),
            "AP50": mAP[50],
            "AP75": mAP[75],
        }

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {
                    "bAP": np.mean(list(mAP_base.values())),
                    "bAP50": mAP_base[50],
                    "bAP75": mAP_base[75],
                }
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update(
                {
                    "nAP": np.mean(list(mAP_novel.values())),
                    "nAP50": mAP_novel[50],
                    "nAP75": mAP_novel[75],
                }
            )

        # write per class AP to logger
        per_class_res = {
            self._class_names[idx]: ap for idx, ap in enumerate(aps[50])
        }

        self._logger.info(
            "Evaluate per-class mAP50:\n" + str(per_class_res)
        )
        self._logger.info(
            "Evaluate overall bbox:\n" + str(ret["bbox"])
        )
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a SDAC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

def sdac_ap(rec, prec) -> float:

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def sdac_eval(
    detpath,
    base_annotations_directory,
    base_images_directory,
    classname,
    ovthresh=0.5
): 
    print(f'Evaluating {classname} with threshold: {ovthresh}...')
    imagenames = []
    image_ids = []
    # load annots
    recs = {}
    for imagename in os.listdir(base_images_directory):
        image_id = imagename.split(".")[0]
        imagenames.append(imagename)
        image_ids.append(image_id)
        annotation_file = os.path.join(base_annotations_directory, image_id + ".xml")
        recs[image_id] = parse_rec(annotation_file)

    # extract gt objects for this class
    class_recs = {}

    for image_id in image_ids:
        R = [obj for obj in recs[image_id] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        det = [False] * len(R)
        class_recs[image_id] = {
            "bbox": bbox,
            "det": det,
        }

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()
    
    print(class_recs)

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0].split(".")[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(
        -1, 4
    )

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    print(BB)
    print(image_ids)
    print(imagenames)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        print("-----")
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0)
                * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = sdac_ap(rec, prec)

    return rec, prec, ap