from tqdm import tqdm
import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.common.utils import boxes_to_sensor
import shutil
import os


cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def visualize_sample_bev(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    # boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    # boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    # for box_est, box_est_global in zip(boxes_est, boxes_est_global):
    #     box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    # for box in boxes_est:
    #     # Show only predictions with a high score.
    #     assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
    #     if box.score >= conf_th:
    #         box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def show_lidar_gt(sample_token, s_path):
    bbox_gt_list = []
    # bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except:
            pass

    gt_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)

    # 在bev的视角下显示雷达的数据
    out_path = s_path
    visualize_sample_bev(nusc, sample_token, gt_annotations,  savepath=out_path)


def show_cam_gt(sample_token,
                s_path,
                ax=None,
                box_vis_level: BoxVisibility = BoxVisibility.ANY,
                with_anns: bool = True,
                lidarseg_preds_bin_path: str = None):
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(24, 9))
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            # boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
            #              name=record['detection_name'], token='predicted') for record in
            #          pred_data['results'][sample_toekn] if record['detection_score'] > 0.2]

            # data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
            #                                                              box_vis_level=box_vis_level, pred_anns=boxes)
            data_path = nusc.get_sample_data_path(sample_data_token)

            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            camera_intrinsic = np.array(cs_record['camera_intrinsic'])

            _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)
            # mmcv.imwrite(np.array(data)[:,:,::-1], f'{cam}.png')
            # Init axes.

            # Show image.
            ax[j, ind].imshow(data)
            # ax[j + 2, ind].imshow(data)

            # Show boxes.
            if with_anns:
                # for box in boxes_pred:
                #     c = np.array(get_color(box.name)) / 255.0
                #     box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                for box in boxes_gt:
                    c = np.array(get_color(box.name)) / 255.0
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            # ax[j + 2, ind].set_xlim(0, data.size[0])
            # ax[j + 2, ind].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        # ax[j, ind].axis('off')
        # ax[j, ind].set_title('PRED: {} {labels_type}'.format(
        #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        # ax[j, ind].set_aspect('equal')

        ax[j, ind].axis('off')
        ax[j, ind].set_title('GT:{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax[j, ind].set_aspect('equal')

    out_path = s_path
    if out_path is not None:
        plt.savefig(out_path+'camera', bbox_inches='tight', pad_inches=0, dpi=200)
    # if verbose:
    #     plt.show()
    plt.close()


def show_nuscene_gt(nusc):
    scene_num = len(nusc.scene)
    for i in tqdm(range(scene_num), desc="scenes"):
        my_scene = nusc.scene[i]
        first_sample_token = my_scene['first_sample_token']
        sample_token = first_sample_token
        count = 0

        while sample_token != "":
            count = count + 1
            sample = nusc.get("sample", sample_token)

            s_cams_path = os.path.join(s_debug_root, "cams", "_".join([str(count), sample_token, "cams"]))
            s_pc_path = os.path.join(s_debug_root, "pcs", "_".join([str(count), sample_token, "pc"]))

            show_lidar_gt(sample_token, s_pc_path)
            show_cam_gt(sample_token, s_cams_path)

            print(count)
            sample_token = sample['next']

        exit(1)



if __name__ == "__main__":
    print("Start")
    data_root = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuScenes_mini"
    version = 'v1.0-mini'

    s_debug_root = "/home/dell/liyongjing/data/debug_occ_generate/show_gts"
    if os.path.exists(s_debug_root):
        shutil.rmtree(s_debug_root)
    os.mkdir(s_debug_root)

    os.mkdir(os.path.join(s_debug_root, "cams"))
    os.mkdir(os.path.join(s_debug_root, "pcs"))

    # nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    show_nuscene_gt(nusc)
    print("End")
