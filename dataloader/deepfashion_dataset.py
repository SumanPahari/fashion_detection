from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class deepfashion(Dataset):

    def __init__(self, root_dir, csv_path, transforms=None):
        df_raw = pd.read_csv(csv_path)
        self.df = df_raw.reset_index()[["image_name", "x_1", "y_1", "x_2", "y_2", "category_encoding", "area", "index"]]
        self._transforms = transforms
        self._root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df = self.df.iloc[idx, :]
        dictionary = df.to_dict()
        dictionary

        img_path = os.path.join(self._root_dir, dictionary['image_name'])
        image = Image.open(img_path)

        boxes = np.zeros((1, 4), dtype=np.float32)
        boxes[0, 0] = dictionary['x_1']
        boxes[0, 1] = dictionary['y_1']
        boxes[0, 2] = dictionary['x_2']
        boxes[0, 3] = dictionary['y_2']

        gt_classes = np.zeros(1, dtype=np.int32)
        gt_classes[0] = dictionary['category_encoding']

        area = dictionary['area']

        # convert everything into a torch.Tensor
        image_id = torch.tensor(dictionary['index'], dtype=torch.int32)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
        # area = torch.as_tensor(area, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # target = {"boxes": boxes, "labels": gt_classes, "image_id": image_id, "area": area}
        target = {"boxes": boxes, "labels": gt_classes, "area": area, "image_id": image_id}

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @property
    def class_to_coco_cat_id(self):
        return self._class_to_coco_cat_id
