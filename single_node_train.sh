#!/bin/bash
#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH

# Add project root to PYTHONPATH
export PYTHONPATH="/root/autodl-fs/medsam2:${PYTHONPATH}"
# make sure the checkpoint is under `MedSAM2/checkpoints/sam2.1_hiera_tiny.pt`
config=configs/sam2.1_hiera_tiny_finetune512.yaml
output_path=./exp_log/MedSAM2_TF

# Function to run the training script
CUDA_VISIBLE_DEVICES=0 python training/train_sequential_classes.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 
        # --master-addr $MASTER_ADDR \
        # --main-port $MASTER_PORT

echo "training done"

