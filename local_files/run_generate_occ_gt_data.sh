#cd $Home/tools/generate_occupancy_nuscenes
#python generate_occupancy_nuscenes.py --config_path ./config.yaml --label_mapping ./nuscenes.yaml --split [train/val] --save_path [your/save/path]

work_root="/home/dell/liyongjing/programs/SurroundOcc-liyj/tools/generate_occupancy_nuscenes"
cd $work_root

scene_start_id=4     # scene id
scene_end_id=850
split="all"

echo "scene_start_id:"$scene_start_id
echo "scene_end_id:"$scene_end_id
echo "split:"$split

python generate_occupancy_nuscenes.py \
 --config_path ./config.yaml --label_mapping ./nuscenes.yaml --split val \
 --save_path /home/dell/下载/debug \
 --dataroot /home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuScenes_mini \
 --start $scene_start_id \
 --end $scene_end_id \
 --split $split     # 'train' 'val' 'all'