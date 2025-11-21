# experiments=(
#     #"configs/KD/camvid_DIFF2Seg_512t384s_at_fold1.py"
#     "configs/KD/camvid_DIFF2Seg_512t384s_textkd_innerdot.py"
# )

# echo "Starting experiments..."
# echo "Total experiments: ${#experiments[@]}"

# export CUDA_VISIBLE_DEVICES=0

# for exp in "${experiments[@]}"; do
#     echo "Running experiment: ${exp}"

#     PYTHONPATH=$PYTHONPATH:$(pwd) python tools/train.py "${exp}" 

#     if [ $? -eq 0 ]; then
#         echo "Successfully completed experiment: ${exp}"
#     else
#         echo "Failed to complete experiment: ${exp}"
#         exit 1
#     fi

#     echo "----------------------------------"
# done

# echo "All experiments completed"

# #bash experiments.sh
# # pkill -f "python tools/train.py"
# # nohup bash experiments.sh > crossattkd_lamda_0.001_fold2.log 2>&1 &

experiments=("/home/leeju2/diffseg/IMRL_Project-main/configs/KD/camvid_DIFF2Seg_512t384s_crossatt_kd_fold1_0.001.py")

echo "Starting experiments..."
echo "Total experiments: ${#experiments[@]}"

# ============ 여기서 GPU 설정! ============export CUDA_VISIBLE_DEVICES=0  # GPU 1번 사용
export CUDA_VISIBLE_DEVICES=0  # GPU 1번 사용

# =========================================

for exp in "${experiments[@]}"; do
    echo "Running experiment: ${exp}"
    
    PYTHONPATH=$PYTHONPATH:$(pwd) python tools/train.py "${exp}" 
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed experiment: ${exp}"
    else
        echo "Failed to complete experiment: ${exp}"
        exit 1
    fi
    
    echo "----------------------------------"
done

echo "All experiments completed"