"""
Given provided COCO dataset(imgs and ground truth), fit SPIN model
Then convert the output into a format that is ready for Unity (LHS).

Note: For now only single person images are considered.

sample:
python3 transfer_COCO.py --checkpoint=data/model_checkpoint.pt --annotation=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/annotations/person_keypoints_val2017.json --imgs_folder=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/images/val2017

"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

from models import hmr, SMPL
from utils.imutils import crop
import config
import constants

from tqdm import tqdm
from collections import Counter
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument("--annotation", required=True, help="Path to json file of annotation")
parser.add_argument("--imgs_folder", required=True, help="Path to the folder containing COCO imgs")
parser.add_argument("--output_folder", default="COCO_SPINoutput", help="Path to the folder to put output jsons")

def format_bbox(bbox):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_path, bbox, input_res=224):
    """
    Read image, do preprocessing and crop it according to the bounding box.
    """
    center, scale = bbox
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_path)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # read in COCO annotation json
    COCO_annotations = []
    COCO_imgs = []
    with open(args.annotation, "r") as f:
        COCO_json = json.load(f)
        COCO_annotations = COCO_json["annotations"]
        COCO_imgs = COCO_json["images"]

    # mapping img ids to name
    imgid_name = dict()
    print("Mapping COCO image id to image name...")
    for img in tqdm(COCO_imgs):
        imgid_name[img["id"]] = img["file_name"]

    # select suitable samples
    single_imgids = set()
    crowd_imgids = set()
    for annot in COCO_annotations:
        if annot["category_id"] != 1: continue
        if annot["iscrowd"] == 1: continue
        if annot["num_keypoints"] == 0: continue
        # Kpt Counter({0: 4425, 12: 642, 13: 604, 15: 584, 16: 574, 14: 467, 10: 461, 11: 448,
        # 9: 375, 17: 336, 8: 321, 6: 317, 7: 301, 4: 226, 5: 217, 2: 173, 3: 167, 1: 139})
        if annot["image_id"] in crowd_imgids:
            continue
        elif annot["image_id"] in single_imgids:
            single_imgids.remove(annot["image_id"])
            crowd_imgids.add(annot["image_id"])

    output_json_template = {
        "dataset": "COCO-" + args.imgs_folder.split("/")[-1],
        "name": None,
        "betas": None,
        "poses": None, # expect to be [[w,x,y,z]]
        "camera_trans": None,
        "bbox_center_percent": None, # [x,y] relative to img bottom-left point
    }

    # second pass
    for annot in COCO_annotations:
        if annot["category_id"] != 1: continue
        if annot["iscrowd"] == 1: continue
        if annot["num_keypoints"] == 0: continue

        if annot["image_id"] in single_imgids:
            img_name = imgid_name[annot["image_id"]]
            img_path = os.path.join(args.imgs_folder, img_name)
            formatted_bbox = format_bbox(annot["bbox"])
            img, norm_img = process_image(img_path, formatted_bbox)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
                output_json = dict(output_json_template)
                output_json["name"] = img_name
                output_json["betas"] = pred_betas.cpu().tolist()[0]
                output_json["poses"] = rotMat2Quat(pred_rotmat.cpu().numpy())
                output_json["camera_trans"] = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)[0].cpu().tolist()
                output_json["bbox_center_percent"] = [formatted_bbox[0][0]/w, 1-formatted_bbox[0][1]/h]
#
