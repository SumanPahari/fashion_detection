
class Config:
    # backbone = 'mobilenet'  # [vgg16, resnet-fpn, mobilenet, resnet50_fpn]
    backbone = 'resnet50_fpn'  # [vgg16, resnet-fpn, mobilenet, resnet50_fpn]
    backbone_pretrained_weights = None  # [path or None]
    # backbone_pretrained_weights = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/mobilenet_v2-b0353104.pth"  # [path or None]
    # backbone_pretrained_weights = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"  # [path or None]

    # data transform parameter
    train_horizon_flip_prob = 0.0  # data horizon flip probility in train transform
    min_size = 800
    max_size = 1000
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # anchor parameters
    # anchor_size = [64, 128, 256]
    anchor_size = [32, 64, 128, 256, 512]
    anchor_ratio = [0.5, 1, 2.0]

    # roi align parameters
    roi_out_size = [7, 7]
    roi_sample_rate = 2

    # rpn process parameters
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 2000

    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 1000

    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    # remove low threshold target
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None

    # device_name = 'cuda:7'
    device_name = 'cpu'

    resume = ''  # pretrained_weights
    start_epoch = 0  # start epoch
    num_epochs = 20  # train epochs
    print_freq = 5  # train epochs

    # learning rate parameters
    lr = 5e-3
    momentum = 0.9
    weight_decay = 0.0005

    # learning rate schedule
    lr_gamma = 0.33
    lr_dec_step_size = 100

    # batch_size = 6
    batch_size = 6

    num_class = 10 + 1  # foreground + 1 background
    data_root_dir = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/small_data/"
    model_save_dir = "/Users/sumanpahari/Documents/Data Science/Thesis/Datasets/Category and Attribute Prediction Benchmark/pytorch-faster-rcnn/model"


cfg = Config()
