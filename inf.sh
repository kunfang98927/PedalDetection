#!/bin/bash
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=8000MB
#SBATCH --time=0-02:00:00
#SBATCH --output=train-%N-%j.out
#SBATCH --error=train-%N-%j.err
#SBATCH --job-name=inf-batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kun.fang@mail.mcgill.ca
#SBATCH --account=def-ichiro

nvidia-smi
module load arrow/15.0.1 StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13 git-lfs/3.4.0 rust/1.76.0
source env/bin/activate

pip list


# List of checkpoint paths
checkpoints=(
  # "ckpt-0319_real_onoff/model_epoch_3_step_42000_val_loss_0.0490_f1_0.8713_mae_0.1427.pt"
  # "ckpt-0319_real+r1pf1+r2pf1_onoff/model_epoch_1_step_33000_val_loss_0.0452_f1_0.7740_mae_0.1288.pt"
  # "ckpt-0319_real+r1pf1+r3pf1_onoff/model_epoch_1_step_36000_val_loss_0.0503_f1_0.8417_mae_0.1430.pt"
  # "ckpt-0319_real+r2pf1+r3pf1_onoff/model_epoch_1_step_33000_val_loss_0.0490_f1_0.8873_mae_0.1464.pt"
  # "ckpt-0319_real+r1pf1+r2pf1+r3pf1_onoff/model_epoch_1_step_24000_val_loss_0.0505_f1_0.8366_mae_0.1390.pt"

  # "ckpt-0319_real_onoff/model_epoch_5_step_66000_val_loss_0.0475_f1_0.8383_mae_0.1342.pt"
  # "ckpt-0319_real+r1pf1+r2pf1_onoff/model_epoch_2_step_48000_val_loss_0.0414_f1_0.8581_mae_0.1219.pt"
  # "ckpt-0319_real+r1pf1+r3pf1_onoff/model_epoch_2_step_54000_val_loss_0.0476_f1_0.7880_mae_0.1335.pt"
  # "ckpt-0319_real+r2pf1+r3pf1_onoff/model_epoch_2_step_51000_val_loss_0.0474_f1_0.8907_mae_0.1397.pt"
  "ckpt-0319_real+r1pf1+r2pf1+r3pf1_onoff/model_epoch_1_step_44000_val_loss_0.0479_f1_0.8372_mae_0.1387.pt"
)

# List of datasets
datasets=(
  "r0-pf1"
  "r1-pf1"
  "r2-pf1"
  "r3-pf1"
  "r4-pf1"
  "r5-pf1"
  "r1-pf0"
  "r2-pf0"
  "r3-pf0"
)

# Loop over each checkpoint and dataset
for checkpoint in "${checkpoints[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running inference on dataset ${dataset} with checkpoint ${checkpoint}"
    python inference_batch.py --dataset "${dataset}" --checkpoint_path "${checkpoint}"
  done
done