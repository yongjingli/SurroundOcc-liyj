# nuscences 数据下载介绍
#https://www.nuscenes.org/download

# 需要注意的是,采用的是mmdetection3d的v0.17.1版本,需要采用该版本的create_data.py
# 采用不同版本create_data.py产生的infos_xxx.pkl是不一样的
# 生成完后可能需要修改下名字，nuscenes_infos_temporal_train.pk修改为nuscenes_infos_train.pk
cd ../
python tools/create_data.py nuscenes --root-path ./data/nuScenes_mini --version v1.0-mini --out-dir ./data --extra-tag nuscenes