WEIGHT=1000
EXP_NAME=lafan_90
DATA_PATH=../datasets/LAFAN1/output/BVH
SKELETON_PATH=${DATA_PATH}/jumps1_subject2.bvh

python run_cmib.py \
    --weight ${WEIGHT} \
    --exp_name ${EXP_NAME} \
    --data_path ${DATA_PATH} \
    --skeleton_path ${SKELETON_PATH} \
    --save_path tests/ \
    --plot_image=True
