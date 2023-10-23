import open3d as o3d
import os
import sys
import pdb
import time
import yaml
# import open3d
import torch
import chamfer
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation

# import open3d
# import open3d as o3d
from copy import deepcopy
import shutil

from debug_utils import ncolors, save_pts_with_colors

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


RGBS = ncolors(20)
RGBS = color_list + RGBS


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances




def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc


def main(nusc, val_list, indice, nuscenesyaml, args, config):

    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if args.split == 'train':
        if my_scene['token'] in val_list:
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:
            return
    elif args.split == 'all':
        pass
    else:
        raise NotImplementedError


    # load the first sample to start
    # 一个scene包含很多的sample，类似一个视频有不同的帧组成
    first_sample_token = my_scene['first_sample_token']

    # nusc.get的作用"Table record. See README.md for record details for each table."
    # 得到scene里第一个sample，类似得到视频里的第一帧
    my_sample = nusc.get('sample', first_sample_token)

    # 对于第一个sample，得到该时刻的传感器数据，sensor为传感器类型，如'LIDAR_TOP'和CAM_BACK等
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])

    # 得到传感器（sample_data）的数据后, 可以得到传感器到ego的RT
    # 包含关于自车相对于全局坐标系的位置(translation编码)和方向(rotation编码) 的信息
    # 用于将车身坐标转到全局坐标
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 定义在特定车辆上校准的特定传感器(激光雷达/雷达/摄像机)。所有外部参数都是关于自我车体框架给出的。所有相机图像都没有失真和校正。
    # 用于将车身坐标转到传感器坐标
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence
    dict_list = []

    # 用来记录处理的数据的帧号
    frame_num = 0
    DEBUG = True

    # 终止条件为遍历该场景的所有sample
    while True:
        ############################# get boxes ##########################
        # "Returns the data path as well as all annotations related to that sample_data."
        # 是在世界坐标里进行的标注, 理论上所有的点云都会有相关的标注信息
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            # category = human.pedestrian.adult
            for (j, label) in enumerate(nuscenesyaml['labels']):
                # label=2
                # 2: 'human.pedestrian.adult,
                if category == nuscenesyaml['labels'][label]:
                    # append, 7
                    # 2: 7
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename'] # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        if lidar_data['is_key_frame']: # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
        object_points_list = []
        j = 0
        # points_in_boxes [1 34688, 69], 34688代表点云的数量，69代表box的数量
        # object_points_list 代表每个物体box包含的点云有哪些
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:,j].bool()
            object_points = pc0[object_points_mask]
            object_points_list.append(object_points)
            j = j + 1

        # points_mask代表该点云是否在boxes内
        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        #  config['self_range']=[3.0, 3.0, 3.0]
        #  将属于自身车辆本身的点云扣除
        ############################# get point mask of the vehicle itself ##########################
        range = config['self_range']
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                        (np.abs(pc0[:, 1]) > range[1]) |
                                        (np.abs(pc0[:, 2]) > range[2]))

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        # 需要注意的是，由于每张图像的时间戳、激光的时间戳都两两不相同，它们有各自的位姿补偿（ego data），进行坐标系转换的时候需要注意一下。
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        # 先转到世界坐标（雷达坐标先转到车身坐标，然后转到全局坐标），然后在转到第一帧的lidar坐标（全局坐标转到车身坐标，然后再转到第一帧的雷达坐标）
        lidar_pc = lidar_to_world_to_lidar(pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                           lidar_calibrated_sensor0,
                                           lidar_ego_pose0)

        # print(lidar_calibrated_sensor==lidar_calibrated_sensor0, lidar_ego_pose==lidar_ego_pose0)
        # 输出的结果为[True, False],不同帧时，lidar_calibrated_sensor是一样的，车身到具体传感器（雷达）
        # 但是ego_pose是不一样的，该ego_pose就是将ego坐标转到global坐标的

        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens,
                "object_points_list": object_points_list,
                "lidar_pc": lidar_pc.points,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "gt_bbox_3d": gt_bbox_3d,
                "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1]}
        ################## record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame']:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                             lidar_calibrated_sensor.copy(),
                                                             lidar_ego_pose.copy(),
                                                             lidar_calibrated_sensor0,
                                                             lidar_ego_pose0)
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(dict)

        if DEBUG:
            # 用来进行debug的保存
            frame_num = frame_num + 1
            name = lidar_data['token']

            # 保存去除物体前后的点云
            # pc0 为原始的点云， pc为去除物体后的点云， lidar_pc为转到第一帧坐标的去除物体后的点云
            s_pc_path = os.path.join(s_debug_root, "pc0", "_".join([str(frame_num),name,  "pc0"]))
            save_ply(pc0, s_pc_path)

            s_pc_path = os.path.join(s_debug_root, "pc", "_".join([str(frame_num),name,  "pc"]))
            save_ply(pc, s_pc_path)

            s_pc_path = os.path.join(s_debug_root, "lidar_pc", "_".join([str(frame_num), name, "lidar_pc"]))
            save_ply(lidar_pc.points.transpose(1, 0), s_pc_path)


        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next']

        if next_token != '':
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break

        # TODO
        # break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    for dict in dict_list:
        if dict['is_key_frame']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic'])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T

    if DEBUG:
        label = lidar_pc_with_semantic[:, 3]
        colors = np.array(RGBS)[label.astype(np.int32)]
        s_pc_path = os.path.join(s_debug_root, "lidar_pc_with_semantic.ply")
        save_pts_with_colors(s_pc_path, lidar_pc_with_semantic, colors)


    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)
                else:
                    continue

        if DEBUG:
            for i, object_points in enumerate(object_points_dict[query_object_token]):
                s_pc_path = os.path.join(s_debug_root, "objects", "_".join([query_object_token, str(i)]))
                save_ply(object_points, s_pc_path)

        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0)

    object_points_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:,:3])
    # print('object finish')

    # exit(1)
    i = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print('finish scene!')
            return
        dict = dict_list[i]
        is_key_frame = dict['is_key_frame']
        if not is_key_frame: # only use key frame as GT
            i = i + 1
            continue

        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']
        lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
                                                      lidar_calibrated_sensor0.copy(),
                                                      lidar_ego_pose0.copy(),
                                                      lidar_calibrated_sensor,
                                                      lidar_ego_pose)
        point_cloud = lidar_pc_i.points.T[:,:3]
        point_cloud_with_semantic = lidar_pc_i_semantic.points.T

        ################## load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        rots = gt_bbox_3d[:,6:7]
        locs = gt_bbox_3d[:,0:3]

        ################## bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token==object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                        points = points[points_in_boxes[0,:,0].bool()]

                    object_points_list.append(points)
                    semantics = np.ones_like(points[:,0:1]) * object_semantic[k]
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try: # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp])
        except:
            scene_points = point_cloud
        try:
            temp = np.concatenate(object_semantic_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
        except:
            scene_semantic_points = point_cloud_with_semantic


        if DEBUG:
            # scenes
            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_object" + str(i))
            save_ply(np.concatenate(object_points_list), s_pc_path)

            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_static" + str(i))
            save_ply(point_cloud, s_pc_path)

            label = scene_semantic_points[:, 3]
            colors = np.array(RGBS)[label.astype(np.int32)]
            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_sematic" + str(i) + ".ply")
            save_pts_with_colors(s_pc_path, scene_semantic_points, colors)

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
               & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0)
        scene_points = scene_points[mask]

        if DEBUG:
            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_all_mask" + str(i))
            save_ply(scene_points, s_pc_path)

        ################## get mesh via Possion Surface Reconstruction ##############
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
        with_normal = preprocess(point_cloud_original, config)
        with_normal2.points = with_normal.points
        with_normal2.normals = with_normal.normals
        mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                       config['min_density'], with_normal2)

        scene_points = np.asarray(mesh.vertices, dtype=float)

        if DEBUG:
            # 保存mesh模型
            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_mesh" + str(i) + ".ply")
            o3d.io.write_triangle_mesh(s_pc_path, mesh)

            s_pc_path = os.path.join(s_debug_root, "scenes", "scene_possion" + str(i) + ".ply")
            save_ply(scene_points, s_pc_path)

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
               & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0)
        scene_points = scene_points[mask]

        ################## convert points to voxels ##############
        pcd_np = scene_points
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        pcd_np = np.floor(pcd_np).astype(np.int)
        voxel = np.zeros(occ_size)
        voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

        ################## convert voxel coordinates to LiDAR system  ##############
        gt_ = voxel
        x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
        y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
        z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        vv = np.stack([X, Y, Z], axis=-1)
        fov_voxels = vv[gt_ > 0]
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        ################## get semantics of sparse points  ##############
        mask = (np.abs(scene_semantic_points[:, 0]) < 50.0) & (np.abs(scene_semantic_points[:, 1]) < 50.0) \
               & (scene_semantic_points[:, 2] > -5.0) & (scene_semantic_points[:, 2] < 3.0)
        scene_semantic_points = scene_semantic_points[mask]

        ################## Nearest Neighbor to assign semantics ##############
        dense_voxels = fov_voxels
        sparse_voxels_semantic = scene_semantic_points

        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
        y = torch.from_numpy(sparse_voxels_semantic[:,:3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer.forward(x,y)
        indices = idx1[0].cpu().numpy()


        dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
        dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

        # to voxel coordinate
        pcd_np = dense_voxels_with_semantic
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int)

        dirs = os.path.join(save_path, 'dense_voxels_with_semantic/')
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        np.save(os.path.join(dirs, dict['pc_file_name'] + '.npy'), dense_voxels_with_semantic)

        if DEBUG:
            s_pc_path = os.path.join(s_debug_root, "sematic_voxel", "voxels_with_semantic" + str(i) + ".npy")
            np.save(s_pc_path, dense_voxels_with_semantic)

        i = i + 1
        continue


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='./data/GT_occupancy/')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=850)
    # parse.add_argument('--dataroot', type=str, default='./data/nuScenes/')
    parse.add_argument('--dataroot', type=str, default='./data/nuScenes_mini/')
    parse.add_argument('--nusc_val_list', type=str, default='./nuscenes_val_list.txt')
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml')
    args=parse.parse_args()

    s_debug_root = "/home/dell/liyongjing/data/debug_occ_generate/occ_generate"
    if os.path.exists(s_debug_root):
        shutil.rmtree(s_debug_root)
    os.mkdir(s_debug_root)

    os.mkdir(os.path.join(s_debug_root, "pc"))
    os.mkdir(os.path.join(s_debug_root, "pc0"))
    os.mkdir(os.path.join(s_debug_root, "lidar_pc"))
    os.mkdir(os.path.join(s_debug_root, "objects"))
    os.mkdir(os.path.join(s_debug_root, "scenes"))
    os.mkdir(os.path.join(s_debug_root, "sematic_voxel"))

    if args.dataset=='nuscenes':
        val_list = []
        with open(args.nusc_val_list, 'r') as file:
            for item in file:
                val_list.append(item[:-1])
        file.close()

        # nusc = NuScenes(version='v1.0-trainval',
        #                 dataroot=args.dataroot,
        #                 verbose=True)
        nusc = NuScenes(version='v1.0-mini',
                        dataroot=args.dataroot,
                        verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        raise NotImplementedError

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)

    for i in range(args.start,args.end):
        # i = 4
        print('processing sequecne:', i)
        if i < len(nusc.scene):
            main(nusc, val_list, indice=i,
                 nuscenesyaml=nuscenesyaml, args=args, config=config)
        else:
            print("skip {}, max length of scenes is {}".format(i, len(nusc.scene)))

        exit(1)
