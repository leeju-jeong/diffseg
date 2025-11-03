experiments=(
    #"configs/KD/camvid_DIFF2Seg_512t384s_at_fold1.py"
    "configs/KD/camvid_DIFF2Seg_512t384s_textkd_innerdot.py"
)

echo "Starting experiments..."
echo "Total experiments: ${#experiments[@]}"

export CUDA_VISIBLE_DEVICES=0

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

#bash experiments.sh
# pkill -f "python tools/train.py"
# nohup bash experiments.sh > txtkd_lamd1_fold1_classidx.log 2>&1 &

