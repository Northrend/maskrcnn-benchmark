from __future__ import print_function
import os
import argparse
import pprint
import time
import json
import cv2
import torch

from torchvision import transforms as T
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from predictor import COCODemo


class DummyDemo(COCODemo):
    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        # show_mask_heatmaps=False,
        # masks_per_dim=2,
        min_image_size=224,
        dataset_name = None
    ):
        assert dataset_name, 'Dataset name is required'
        if dataset_name == 'blued':
            self.CATEGORIES = ['__background__', 'beard', 'black_frame_glasses', 'police_cap', 'sun_glasses', 'stud_earrings', 'mouth_mask', 'bangs', 'tattoo', 'shirt', 'suit', 'tie', 'belt', 'jeans', 'shorts', 'leg_hair', 'military_uniform', 'under_shirt', 'gloves', 'pecs', 'abdominal_muscles', 'calf', 'briefs', 'boxers', 'butt', 'leather_shoes', 'black_socks', 'white_socks', 'feet', 'non_leather_shoes', 'hot_pants']
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.TEST.TEST_WEIGHT)

        self.transforms = self.build_transform()

        # mask_threshold = -1 if show_mask_heatmaps else 0.5
        # self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        # self.show_mask_heatmaps = show_mask_heatmaps
        self.show_mask_heatmaps = False 
        self.masks_per_dim = 2 

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        filtered_pred = self.select_top_predictions(predictions)
        scores = filtered_pred.get_field("scores").tolist()
        labels = [self.CATEGORIES[i] for i in filtered_pred.get_field("labels").tolist()]
        boxes = filtered_pred.bbox
        result_list = list()
        for box, score, label in zip(boxes, scores, labels):
            # export boxes as [x1,y1,x2,y2,score,class]
            temp_box = [round(float(x),6) for x in box]
            temp_box.append(round(float(score),6))
            temp_box.append(label)
            result_list.append(temp_box)
        return result_list

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Dummy Dataset Demo")
    parser.add_argument(
        "--input-file",
        default="test.jpg",
        metavar="FILE",
        help="path to input image/list file",
    )
    parser.add_argument(
        "--output-json",
        default="test_result.json",
        metavar="FILE",
        help="path to output json file",
    )
    parser.add_argument(
        "--img-prefix",
        default="",
        metavar="PATH",
        help="path to prefix of input images",
    )
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset",
        default="blued",
        metavar="STR",
        help="dataset name, choose from [blued,]",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    dummy_demo = DummyDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        # show_mask_heatmaps=args.show_mask_heatmaps,
        # masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        dataset_name = args.dataset
    )

    start_time = time.time()
    if args.input_file.endswith('lst'): # list of images inference
        result_dict = dict()
        with open(args.input_file,'r') as f:
            if args.img_prefix:
                img_list = [os.path.join(args.img_prefix, x.strip()) for x in f.readlines()]
            else:
                img_list = [x.strip() for x in f.readlines()]
            for idx,img_name in enumerate(img_list):
                img = cv2.imread(img_name)
                result_list = dummy_demo.run_on_opencv_image(img)
                result_dict[os.path.basename(img_name)] = result_list
                print('==> image[{}]: {}'.format(idx, args.input_file))
                for box in result_list:
                    pprint.pprint(box)
        with open(args.output_json,'w') as f:
            json.dump(result_dict,f,indent=2)
        print("Results saved as", args.output_json)
        print("Time: {:.4f} s / img".format(float(time.time() - start_time) / len(img_list)))
    else:   # single image inference
        img = cv2.imread(args.input_file)
        result_list = dummy_demo.run_on_opencv_image(img)
        print('==> image[0]:', args.input_file)
        for box in result_list:
            pprint.pprint(box)
        print("Time: {:.4f} s / img".format(time.time() - start_time))

if __name__ == '__main__':
    main()
