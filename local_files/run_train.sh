nvcc -V
#cd ../

#gpu_num=1
work_dirs="./work_dirs/surroundocc"
# ./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py $gpu_num $work_dirs

# 不能采用sh run_infer.sh的方式运行,需要在终端里运行，或者在pytroch中采用module的方式运行
#./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 1 ./work_dirs/surroundocc

# 修改为单gpu训练
python ./tools/train.py ./projects/configs/surroundocc/surroundocc.py --work-dir $work_dirs