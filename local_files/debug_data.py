import os
from tqdm import tqdm
import numpy as np
import mmcv
import cv2
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes


def debug_load_ann_file():
    # ann_file = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/data_info_v1.0-trianval/nuscenes_infos_val.pkl"
    ann_file = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuscenes_infos_temporal_val.pkl"
    load_interval = 1
    data = mmcv.load(ann_file)
    print(data.keys())
    # print(data['data_list'][0])
    # print(data['infos'][0])
    # exit(1)

    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # data_infos = list(sorted(data['meta_infos'], key=lambda e: e['timestamp']))
    data_infos = data_infos[::load_interval]
    metadata = data['metadata']
    version = metadata['version']
    print(version)
    # print(data_infos)

def load_annotations(ann_file):
    data = mmcv.load(ann_file)
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # data_infos = data_infos[::self.load_interval]
    # self.metadata = data['metadata']
    # self.version = self.metadata['version']
    return data_infos
    # return mmcv.load(ann_file)


def checkout_occ_path():
    # ann_file = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuscenes_infos_train.pkl"
    ann_file ='/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuscenes_infos_train.pkl'

    print(mmcv.__version__)
    # data_infos = mmcv.load(ann_file)
    data_infos = load_annotations(ann_file)
    print("fffff", type(data_infos))
    info = data_infos[0]
    occ_path = info['occ_path']
    # occ_size = np.array(self.occ_size)
    # pc_range = np.array(self.pc_range)
    print(occ_path)


def modify_occ_path():

    occ_path = "./data/nuScenes_mini/samples/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin"
    occ_path = os.path.split(occ_path)[0] + "/dense_voxels_with_semantic/" + os.path.split(occ_path)[1] + ".npy"

    # occ的读取方式也是npy
    # occ = np.load(results['occ_path'])
    # occ = occ.astype(np.float32)



def debug_nuscene_id():
    dataroot = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuScenes_mini"
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=dataroot,
                    verbose=True)

    # nusc.scene 为list, mini-nuscene的长度为10
    indice = 0
    my_scene = nusc.scene[indice]

    print("Ed")


def imgs_hconcat(img1, img2):
    img_h1, img_w1, _ = img1.shape
    img_h2, img_w2, _ = img2.shape

    scale_h = img_h1/img_h2
    img2_res_h, img2_res_w = int(scale_h * img_h2), int(scale_h * img_w2)

    img2_res = cv2.resize(img2, (img2_res_w, img2_res_h))
    img_hconcat = cv2.hconcat([img1, img2_res])
    return img_hconcat


def imgs_vconcat(img1, img2):
    img_h1, img_w1, _ = img1.shape
    img_h2, img_w2, _ = img2.shape

    scale_w = img_w1/img_w2
    img2_res_h, img2_res_w = int(scale_w * img_h2), int(scale_w * img_w2)

    img2_res = cv2.resize(img2, (img2_res_w, img2_res_h))
    img_vconcat = cv2.vconcat([img1, img2_res])
    return img_vconcat


def concat_occ_imgs():
    cam_root = "/home/dell/liyongjing/data/debug_occ_generate/show_gts/cams"
    pc_root = "/home/dell/liyongjing/data/debug_occ_generate/show_gts/pcs"
    occ_root = "/home/dell/liyongjing/data/debug_occ_generate/occ_generate/sematic_voxel"
    # s_root = "/home/dell/liyongjing/documents/20230830/occ_imgs"
    s_root = "/home/dell/liyongjing/documents/20230830/box_imgs"
    cam_names = [name for name in os.listdir(cam_root) if name.endswith(".png")]
    pc_names = [name for name in os.listdir(pc_root) if name.endswith(".png")]
    occ_names = [name for name in os.listdir(occ_root) if name.endswith(".png")]

    cam_names = sorted(cam_names, key=lambda x: int(x.split("_")[0]))
    pc_names = sorted(pc_names, key=lambda x: int(x.split("_")[0]))
    occ_names = sorted(occ_names, key=lambda x: int(x.split(".")[0].split("semantic")[-1]))

    for cam_name, pc_name, occ_name in tqdm(zip(cam_names, pc_names, occ_names)):
        cam_path = os.path.join(cam_root, cam_name)
        pc_path = os.path.join(pc_root, pc_name)
        occ_path = os.path.join(occ_root, occ_name)

        cam_img = cv2.imread(cam_path)
        pc_img = cv2.imread(pc_path)
        occ_img = cv2.imread(occ_path)
        occ_img = occ_img[200:800, 400:1500, :]

        img_hconcat = imgs_hconcat(cam_img, pc_img)
        img_vconcat = imgs_vconcat(occ_img, img_hconcat)

        s_path = os.path.join(s_root, cam_name)
        cv2.imwrite(s_path, img_hconcat)
        # cv2.imwrite(s_path, img_vconcat)


        # plt.imshow(img_vconcat[:, :, ::-1])
        # plt.show()
        # exit(1)


if __name__ == "__main__":
    print("Start")
    # debug_load_ann_file()

    # 在train的时候读取数据路径报错
    # FileNotFoundError: [Errno 2] No such file or directory: \
    #     './data/nuScenes_mini/samples/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800382897959.pcd.bin'
    #
    # checkout_occ_path()

    # /home/dell/liyongjing/programs/SurroundOcc-liyj/projects/mmdet3d_plugin/datasets/nuscenes_occupancy_dataset.py
    # modify_occ_path()

    # 在生成真值的时候报错
    #   File "generate_occupancy_nuscenes.py", line 140, in main
    #     my_scene = nusc.scene[indice]
    # IndexError: list index out of range
    # debug_nuscene_id()

    # 将生成的occ真值图像拼接在一起
    concat_occ_imgs()

    print("End")
