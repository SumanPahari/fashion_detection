class Config:
    model_weights = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/model/resnet50_fpn-model-13-mAp-0.9009899497032166.pth"
    image_path = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/small_data/img/Panda-Hood_Pajama_Onesie/img_00000070.jpg"
    gpu_id = '2'
    num_classes = 5 + 1
    # data_root_dir = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/small_data/img/"
    data_root_dir = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/small_data/"
    save_test_image_path = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/model/test.jpg"

test_cfg = Config()
