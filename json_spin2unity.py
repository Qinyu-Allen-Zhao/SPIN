import os
import json
import pandas as pd
import numpy as np

def rotMat2Quat(poseRotMat):
    res = []

    for i, mat3x3 in enumerate(poseRotMat):
        qw = qx = qy = qz = None

        tr = mat3x3[0][0] + mat3x3[1][1] + mat3x3[2][2]
        if (tr > 0):
            # print("!1!")
            S = np.sqrt(tr + 1) *2
            qw = 0.25*S
            qx = (mat3x3[2][1] - mat3x3[1][2]) / S
            qy = (mat3x3[0][2] - mat3x3[2][0]) / S
            qz = (mat3x3[1][0] - mat3x3[0][1]) / S
        elif ((mat3x3[0][0] > mat3x3[1][1]) & (mat3x3[0][0] > mat3x3[2][2])):
            # print("!2!")
            S = np.sqrt(1 + mat3x3[0][0] - mat3x3[1][1] - mat3x3[2][2]) * 2
            qw = (mat3x3[2][1] - mat3x3[1][2]) / S
            qx = 0.25*S
            qy = (mat3x3[1][0] + mat3x3[0][1]) / S
            qz = (mat3x3[2][0] + mat3x3[0][2]) / S
        elif ((mat3x3[1][1] > mat3x3[2][2])):
            # print("!3!")
            S = np.sqrt(1 + mat3x3[1][1] - mat3x3[0][0] - mat3x3[2][2]) * 2
            qw = (mat3x3[0][2] - mat3x3[2][0]) / S
            qx = (mat3x3[1][0] + mat3x3[0][1]) / S
            qy = 0.25*S
            qz = (mat3x3[2][1] + mat3x3[1][2]) / S
        else:
            # print("!4!")
            S = np.sqrt(1 + mat3x3[2][2] - mat3x3[0][0] - mat3x3[1][1]) * 2
            qw = (mat3x3[1][0] - mat3x3[0][1]) / S
            qx = (mat3x3[2][0] + mat3x3[0][2]) / S
            qy = (mat3x3[2][1] + mat3x3[1][2]) / S
            qz = 0.25*S
        # negate x and w
        # RHS coordinate system to LHS
        res.append([-qw, -qx, qy, qz])
    return res

def output_pretty(spin_output_path):
    # prepare spin json for unity
    # 1. rotation matrix to Quaterion
    # 2. pretty json
    spin_output = dict()
    with open(spin_output_path, "r") as f:
        spin_output = json.load(f)
    res = {
        "dataset": spin_output["dataset:"],
        "name": spin_output["name"],
        "betas": spin_output["pred_betas"][0],
        "poses": None,
        "camera_trans": spin_output["camera_translation"],
    }

    pose_parms = rotMat2Quat(np.concatenate((spin_output["global_orient"][0], spin_output["body_pose"][0]), axis=0))
    res["poses"] = pose_parms
    print(np.array(pose_parms).shape)
    return res

img_name = "000000000785"
path_spin = "./COCO_2017kpt/{}_output.json".format(img_name)
path_unity = "./unity_format_data/{}_spin.json".format(img_name)

json_unity = output_pretty(path_spin)
with open(path_unity, "w") as f:
    json.dump(json_unity, f, indent=4)
