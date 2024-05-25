import torch, detectron2
import numpy as np
import os, json, cv2, random
import time
import sys

# import some common detectron2 utilities
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.modeling.utils import reset_cls_test
from detic.predictor import AsyncPredictor

import pyrealsense2 as rs

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}


class myVisualizationDemo(object):
    def __init__(self, cfg,
                 instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            BUILDIN_METADATA_PATH['lvis'])
        classifier = BUILDIN_CLASSIFIER['lvis']

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


def realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        predictions, visualized_output = demo.run_on_image(color_image)
        cv2.imshow("result", visualized_output.get_image()[:, :, ::-1])
        cv2.waitKey(1)


if __name__ == "__main__":
    # img = cv2.imread("desk.jpg")

    cfg = get_cfg()
    print("1")
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(
        "configs/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.yaml"
    )
    cfg.MODEL.WEIGHTS = "models/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
        False  # For better visualization purpose. Set to False for all classes.
    )
    # cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
    #    "/datasets/metadata/lvis_v1_train_cat_info.json"
    # )
    cfg.freeze()
    print("2")
    demo = myVisualizationDemo(cfg)
    print("3")
    # predictions, visualized_output = demo.run_on_image(img)

    # cv2.imshow("result", visualized_output.get_image()[:, :, ::-1])
    # cv2.waitKey(0)

    realsense()
