import os
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config.test_config import test_cfg
from dataloader.coco_dataset import coco
from dataloader.deepfashion_dataset import deepfashion
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model


def test():
    model = create_model(num_classes=test_cfg.num_classes)

    model.cpu()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])

    # test_data_set = coco(test_cfg.data_root_dir, 'test', '2017', data_transform)
    test_data_set = deepfashion(test_cfg.data_root_dir,
                                '/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/small_data/prepared_5_cat_val.csv',
                                data_transform)

    print("test_data_set", test_data_set)

    # category_index = test_data_set.index

    # index_category = dict(zip(category_index.values(), category_index.keys()))
    #index_category = dict(zip((0, 1), ("Gauchos", "Jodhpurs")))
    index_category = dict(zip((0, 1, 2, 3, 4), ("Gauchos", "Jodhpurs", "Caftan", "Onesie", "Capris")))

    original_img = Image.open(test_cfg.image_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        predictions = model(img.cpu())[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        print(predictions)

        if len(predict_boxes) == 0:
            print("No target detected!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 index_category,
                 # thresh=0.3,
                 thresh=0.39,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        # save a image using extension
        original_img.save(test_cfg.save_test_image_path)


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()
