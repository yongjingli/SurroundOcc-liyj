#First, you need to generate prediction results. Here we use whole validation set as an example.

#cp ./data/nuscenes_infos_val.pkl ./data/infos_inference.pkl
#./tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ./ckpts/surroundocc.pth  1

# 采用single gpu
#python /home/dell/liyongjing/programs/SurroundOcc-liyj/tools/test.py \
#./projects/configs/surroundocc/surroundocc_inference.py \
#./ckpts/surroundocc.pth --format-only

# 不能采用sh run_infer.sh的方式运行,需要在终端里运行，或者在pytroch中采用module的方式运行
cd ../
sh ./tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference.py ./ckpts/surroundocc.pth 1