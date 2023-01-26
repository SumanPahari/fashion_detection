import time
import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.train_utils import MetricLogger
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator


@torch.no_grad()
def evaluate(model, data_loaders, device, mAP_list=None):
    # Need to predict from every data item with boxes,labels and scores
    # create dict pred with predicted boxes,labels and scores
    # create dict with target boxes and labels
    # Then calculate mAP https://github.com/Lightning-AI/metrics/blob/master/examples/detection_map.py

    model = model.to("cpu")
    model.eval()

    # dataiter = iter(data_loaders)
    # images, labels = next(dataiter)
    # predicted = model(images)
    # Initialize metric
    # metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], box_format='xyxy')
    # metric.update(predicted, labels)
    # result = metric.compute()
    # print("Value", result)
    # print("Value", type(result))
    preds = list()
    targets = list()
    for data_loader in iter(data_loaders):
        image, label = data_loader
        predicted = model(image)
        preds.append(predicted[0])
        targets.append(label[0])

        # predicted_d = dict(predicted)
        # print("Predicted_d - ", type(predicted_d))

    # print("Lable - ", label)
    # print("Predicted - ", predicted)

    # Initialize metric
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], box_format='xyxy')

    # Update metric with predictions and respective ground truth
    metric.update(predicted, label)

    # Compute the results
    result = metric.compute()
    print("Value", result)
    # if isinstance(mAP_list, list):
    # mAP_list.append(voc_mAP)

    return None, result.map_50.numpy()
