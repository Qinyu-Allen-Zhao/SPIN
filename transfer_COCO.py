"""
Given provided COCO dataset(imgs and ground truth), fit SPIN model,
Then convert the output into a format that is ready for Unity (LHS).

Note: For now only single person images are considered.

sample:
python3 transfer_COCO.py --checkpoint=data/model_checkpoint.pt --annotation=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/annotations/person_keypoints_val2017.json --imgs_folder=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/images/val2017
python3 transfer_COCO.py --checkpoint=data/model_checkpoint.pt --annotation=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/annotations/person_keypoints_train2017.json --imgs_folder=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/images/train2017

python3 transfer_COCO.py --output_folder=COCO_SPINoutput_crowd --checkpoint=data/model_checkpoint.pt --annotation=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/annotations/person_keypoints_val2017.json --imgs_folder=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/images/val2017 --crowd_imgs
python3 transfer_COCO.py --output_folder=COCO_SPINoutput_crowd --checkpoint=data/model_checkpoint.pt --annotation=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/annotations/person_keypoints_train2017.json --imgs_folder=/home/harold/Desktop/torchProjects/HPE_HRNET/data/coco/images/train2017 --crowd_imgs
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
from json_spin2unity import rotMat2Quat

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument("--annotation", required=True, help="Path to json file of annotation")
parser.add_argument("--imgs_folder", required=True, help="Path to the folder containing COCO imgs")
parser.add_argument("--output_folder", default="COCO_SPINoutput_single", help="Path to the folder to put output jsons")
parser.add_argument("--crowd_imgs", default=False, action="store_true", help="Flag for transferring only crowd images in COCO")

# test_count = 20

def format_bbox(bbox):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    bbox = np.array(bbox)
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

def skip_annot(annot):
    """
    if this annotation should be skipped
    """
    if annot["category_id"] != 1 or annot["iscrowd"] == 1 or annot["num_keypoints"] == 0:
        return True
    return False

if __name__ == '__main__':
    args = parser.parse_args()

    flag_single = not args.crowd_imgs

    # create output folder if needed
    dataset_name = "COCO-" + args.imgs_folder.split("/")[-1]
    if (not os.path.exists(args.output_folder)): os.mkdir(args.output_folder)
    if (not os.path.exists(os.path.join(args.output_folder, dataset_name))): os.mkdir(os.path.join(args.output_folder, dataset_name))

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
    print("\nMapping COCO image id to image name...")
    for img in COCO_imgs:
        imgid_name[img["id"]] = img["file_name"]
    # mapping img ids to dimentions
    imgid_dims = dict()
    print("Mapping COCO image id to image dimention...")
    for img in COCO_imgs:
        imgid_dims[img["id"]] = (img["width"], img["height"])

    # select suitable samples
    print("\nFirst pass: find single imgs and crowd images...")
    single_imgids = set()
    crowd_imgids = set()
    for annot in tqdm(COCO_annotations):
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
        else:
            single_imgids.add(annot["image_id"])
    print("Number single images:",len(single_imgids))
    print("Number crowd images:",len(crowd_imgids))

    output_json_template1 = {
        "name": None,
        "single_person": True,
        "betas": None, # expect to be [b1,b2...b10]
        "poses": None, # expect to be [[w,x,y,z]]
        "camera_trans": None, # expect to be [x,y,z]
        "area": None, # expect to be a
        "bbox_center_percent": None, # [x,y] relative to img bottom-left point
    }

    output_json_template2 = {
        "name": None,
        "single_person": False,
        "person_count": None,
        "betas": None, # expect to be list of [b1,b2...b10]
        "poses": None, # expect to be list of [[w,x,y,z]]
        "camera_trans": None, # expect to be list of [x,y,z]
        "area": None, # expect to be list of a
        "bbox_center_percent": None, # list of [x,y], each [x,y] relative to img bottom-left point
    }

    # second pass
    print("\nSecond pass: feed to SPIN and store output...")
    if flag_single:
        outputs = []
        outputs_json = {
            "dataset": dataset_name,
            "single_person": True,
            "SPIN_outputs": None,
        }
        for annot in tqdm(COCO_annotations):
            if skip_annot(annot): continue
            # print("!")

            if annot["image_id"] in single_imgids:
                img_name = imgid_name[annot["image_id"]]
                img_path = os.path.join(args.imgs_folder, img_name)
                img_name = img_name.split(".")[0] # remove jpg affix
                # if img_name != "000000013201": continue
                formatted_bbox = format_bbox(annot["bbox"])
                # formatted_bbox = format_bbox([0,0,*imgid_dims[annot["image_id"]]])
                img, norm_img = process_image(img_path, formatted_bbox)

                with torch.no_grad():
                    pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
                    w, h = imgid_dims[annot["image_id"]]
                    output_json = dict(output_json_template1)
                    output_json["name"] = img_name
                    output_json["betas"] = pred_betas.cpu().tolist()[0]
                    output_json["poses"] = rotMat2Quat(pred_rotmat[0].cpu().numpy())
                    output_json["camera_trans"] = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)[0].cpu().tolist()
                    output_json["area"] = annot["area"]
                    output_json["bbox_center_percent"] = [formatted_bbox[0][0]/w, 1-formatted_bbox[0][1]/h]
                    outputs.append(output_json)
                    # write output
                    with open(os.path.join(os.path.join(args.output_folder, dataset_name), \
                                img_name+".json"), "w", encoding="utf-8") as f:
                        # print("Writing result json:"+img_name)
                        json.dump(output_json, f, indent=4)
        # a single large file for storing all outputs
        outputs_json["SPIN_outputs"] = outputs
        with open(os.path.join(args.output_folder, \
                    "all_spin_outputs_{}.json".format(dataset_name)), "w", encoding="utf-8") as f:
            json.dump(outputs_json, f, indent=4)
    else:
        outputs_tmpt = dict()
        """Copy from above"""
        for annot in tqdm(COCO_annotations):
            if skip_annot(annot): continue

            if annot["image_id"] in crowd_imgids:
                img_name = imgid_name[annot["image_id"]]
                img_path = os.path.join(args.imgs_folder, img_name)
                img_name = img_name.split(".")[0] # remove jpg affix
                # if img_name != "000000013201": continue
                formatted_bbox = format_bbox(annot["bbox"])
                # formatted_bbox = format_bbox([0,0,*imgid_dims[annot["image_id"]]])
                img, norm_img = process_image(img_path, formatted_bbox)

                with torch.no_grad():
                    pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
                    w, h = imgid_dims[annot["image_id"]]

                    output_json = None
                    if annot["image_id"] in outputs_tmpt.keys():
                        output_json = outputs_tmpt[annot["image_id"]]
                    else:
                        output_json = dict(output_json_template2)
                        for att_name in ["betas", "poses", "camera_trans", "area","bbox_center_percent"]:
                            output_json[att_name] = list()
                        outputs_tmpt[annot["image_id"]] = output_json

                    output_json["name"] = img_name
                    output_json["betas"].append(pred_betas.cpu().tolist()[0])
                    output_json["poses"].append(rotMat2Quat(pred_rotmat[0].cpu().numpy()))
                    output_json["camera_trans"].append(torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)[0].cpu().tolist())
                    output_json["area"].append(annot["area"])
                    output_json["bbox_center_percent"].append([formatted_bbox[0][0]/w, 1-formatted_bbox[0][1]/h])
                    # print(annot["image_id"],img_name)
                    # print(id(output_json))
                    # print(outputs_tmpt.keys(),"\n")
                    # if (len(output_json["betas"]) >= 2): break
        outputs_json = {
            "dataset": dataset_name,
            "single_person": False,
            "max_persons_no": None,
            "SPIN_outputs": [],
        }
        max_persons, max_name = (-1, None)
        for output in outputs_tmpt.values():
            output["person_count"] = len(output["betas"])
            max_persons = max(max_persons, output["person_count"])
            max_name = output["name"] if max_persons == output["person_count"] else max_name

            with open(os.path.join(os.path.join(args.output_folder, dataset_name), \
                        output["name"]+".json"), "w", encoding="utf-8") as f:
                # print("Writing result json:"+img_name)
                json.dump(output, f, indent=4)
            outputs_json["SPIN_outputs"].append(output)
        print("There are at most {} persons in single image, for example {}".format(max_persons, max_name))
        with open(os.path.join(args.output_folder, \
                    "all_spin_outputs_{}.json".format(dataset_name)), "w", encoding="utf-8") as f:
            outputs_json["max_persons_no"] = max_persons
            json.dump(outputs_json, f, indent=4)
#
