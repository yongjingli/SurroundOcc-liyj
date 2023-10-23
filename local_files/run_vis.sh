#You will get prediction results in './visual_dir'. You can directly use meshlab to visualize .ply files or run visual.py to visualize raw .npy files with mayavi:

# 需要注意的是对gt进行可视化的时候需要设置voxel_size = 1.0
# 需要注意的是对预测结果进行可视化的时候需要设置voxel_size = 0.5
# voxel_size = 0.5
# voxel_size = 1.0    # gt

# 结果保存在visual_dir/

#cd ./tools
# pred
#npy_path="/home/dell/liyongjing/programs/SurroundOcc-liyj/visual_dir/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin/pred.npy"

# gt
npy_path="/home/dell/下载/debug/dense_voxels_with_semantic/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin.npy"
python visual.py $npy_path