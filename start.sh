#!/bin/sh -v
# Standard error outputs
#PBS -e /mnt/home/abstrac01/normunds_neimanis/logs
# Stdout
#PBS -o /mnt/home/abstrac01/normunds_neimanis/logs
#PBS -q batch
# ppn: processors per node
#PBS -l nodes=1:ppn=3:gpus=2:shared,feature=k40
# mem: ram size
#PBS -l mem=20gb
#PBS -l walltime=24:00:00
# Sets the name of the job as displayed by qstat
#PBS -N normunds_neimanis
# Merge output and error files
#PBS -j oe

# mem <= 5 * ppn
# -l nodes=1:ppn=8:gpus=1:shared,feature=v100
# -l mem=40gb

# -l nodes=1:ppn=12:gpus=1:shared,feature=k40
# -l mem=60gb
# Multiple GPUs: multiply RAM and CPU count by GPU count
# -l nodes=1:ppn=24:gpus=2:shared,feature=k40
# -l mem=80gb


module load conda
eval "$(conda shell.bash hook)"
conda activate normunds2


# export LD_LIBRARY_PATH=/mnt/home/abstrac01/.conda/envs/normunds_k40/lib/:$LD_LIBRARY_PATH
# echo $LD_LIBRARY_PATH

cd /mnt/home/abstrac01/normunds_neimanis

echo -n "Running on host: "
hostname

#python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --model SotaLSTM --rotate-y --remove-zeromove --center-root --scale --save-artefacts --use-cuda --sequence-name params2 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num -1 &
#python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --save-artefacts --use-cuda --sequence-name params2 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num -1 &
#wait


#python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --model SotaLSTM --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name params2 --epochs 5000 --save-best --overfit-exit-percent 30

# SotaLSTM bestrun
# python taskgen.py --learning-rate 1e-2 --batch-size 128 64 --hidden-size 128 256 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name sotabest --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num 0
# python taskgen.py --learning-rate 1e-3 --batch-size 128 64 --hidden-size 128 256 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name sotabest --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num -1


# Phased LSTM hyperparam search
# --p-lstm-alpha 3e-3 5e-4
# --p-lstm-tau-max 4.0 2.0
# --p-lstm-r-on 8e-2 3e-2

# --p-lstm-alpha 5e-4 3e-4 1e-4 1e-5
# --p-lstm-r-on 0.1 0.15 0.2 0.5

# python taskgen.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 5e-4 --p-lstm-r-on 0.1 0.15 0.2 0.5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name phasedparams --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num 0
# python taskgen.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 3e-4 --p-lstm-r-on 0.1 0.15 0.2 0.5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name phasedparams --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num -1

#python taskgen.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-4 --p-lstm-r-on 0.1 0.15 0.2 0.5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name phasedparams --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num 0
#python taskgen.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 0.15 0.2 0.5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name phasedparams --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 30 --cuda-device-num -1


# Transformer hyperparam search
# python taskgen.py --learning-rate 1e-4 5e-4 --batch-size 64 128 --hidden-size 64 128 --model Transformer --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer2 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num 0
# python taskgen.py --learning-rate 1e-5 1e-6 --batch-size 64 128 --hidden-size 64 128 --model Transformer --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer2 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num -1
## python taskgen.py --learning-rate 1e-4 5e-4 1e-5 1e-6 --batch-size 256 --hidden-size 256 --model Transformer --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer2 --template template_hpc.sh --num-tasks-in-parallel 1 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20 --cuda-device-num -1



# LayerNorm
# python taskgen.py --learning-rate 1e-3 5e-3 --batch-size 64 --hidden-size 64 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num 0
# python taskgen.py --learning-rate 1e-3 5e-3 --batch-size 64 --hidden-size 128 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num -1

#python taskgen.py --learning-rate 1e-3 5e-3 --batch-size 128 --hidden-size 64 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num 0
#python taskgen.py --learning-rate 1e-3 5e-3 --batch-size 128 --hidden-size 128 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num -1

#python taskgen.py --learning-rate 10e-3 --batch-size 128 --hidden-size 64 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num 0
#python taskgen.py --learning-rate 10e-3 --batch-size 128 --hidden-size 128 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num -1

#python taskgen.py --learning-rate 10e-3 --batch-size 128 --hidden-size 64 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num 0
#python taskgen.py --learning-rate 10e-3 --batch-size 128 --hidden-size 128 --model ModelRNN-LayerNorm Model-P-LSTM-LayerNorm SotaLSTM-LayerNorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit2 --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --overfit-exit-percent 20  --cuda-device-num -1



# Hyperparam search and trying to overfit
# 1e-3 3e-3 5e-3 10e-3 20e-3
# python taskgen.py --learning-rate 20e-3 --batch-size 256 --hidden-size 64 128 256 --model ModelRNN Model-P-LSTM SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name overfit --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --cuda-device-num 0
# python ./emotion-gait.py --sequence-name "bestrun" --batch-size 128 --hidden-size 128 --model SotaLSTM --epochs 2000 --learning-rate 0.01 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --cuda-device-num -1 --save-best --overfit-exit-percent 0 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5
# python ./emotion-gait.py --sequence-name "bestrun" --batch-size 64 --hidden-size 64 --model Model-P-LSTM --epochs 1000 --learning-rate 0.003 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --cuda-device-num 0 --save-best --overfit-exit-percent 0 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5
# python ./emotion-gait.py --sequence-name "bestrun" --batch-size 64 --hidden-size 256 --model SotaLSTM --epochs 2000 --learning-rate 0.001 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --cuda-device-num -1 --save-best --overfit-exit-percent 0 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5
# python ./emotion-gait.py --sequence-name "bestrun" --batch-size 128 --hidden-size 64 --model SotaLSTM --epochs 3000 --learning-rate 0.01 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --cuda-device-num 0 --save-best --overfit-exit-percent 0 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5

# SotaLSTM dataset check
#python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 0 &
#sleep 20
#python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 256 --dataset-features features.h5 --dataset-labels labels.h5 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 0 &



# Phased-LSTM dataset check
#sleep 20
#python ./emotion-gait.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 --dataset-features features.h5 --dataset-labels labels.h5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 1 &
#sleep 20
#python ./emotion-gait.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --rotate-y  --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 1 &
#wait


# CVEAGEN dataset check
python ./emotion-gait.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 0 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 256 --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name dataset4 --epochs 5000 --save-best --cuda-device-num 1 &
wait


#### Transformer2
python ./taskgen.py --learning-rate 1e-3 --batch-size 64 128 --hidden-size 64 128 256 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
sleep 20
python ./taskgen.py --learning-rate 1e-4 --batch-size 64 128 --hidden-size 64 128 256 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 &
sleep 20
python ./taskgen.py --learning-rate 1e-5 --batch-size 64 128 --hidden-size 64 128 256 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 2 &
wait


# --regularization-l 3
python ./taskgen.py --learning-rate 1e-2 1e-3 --batch-size 64 128 --hidden-size 32 64 128 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --regularization-l 3 &
wait

# --hidden-size 32
python ./taskgen.py --learning-rate 1e-2 1e-3 --batch-size 64 --hidden-size 32 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
wait



#### Transformer hyperparams --transformer-heads 2 4 12 --transformer-depth 2 4 10 --transformer-embed-dim 32 128
# Transformer not working with 12 heads
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 2  --transformer-depth 2 4 10 --transformer-embed-dim 32 64 128 --regularization-l 0 3 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
sleep 20
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 4 --transformer-depth 2 4 10 --transformer-embed-dim 32 64 128 --regularization-l 0 3 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 &
sleep 20
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 16 --transformer-depth 2 4 10 --transformer-embed-dim 32 64 128 --regularization-l 0 3 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 &
wait

## CVEAGEN dataset check --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 2  --transformer-depth 2 4 10 --transformer-embed-dim 32 128 --regularization-l 0 3 --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3-cvagen --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
sleep 20
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 4 --transformer-depth 2 4 10 --transformer-embed-dim 32 128 --regularization-l 0 3 --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3-cvagen --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 &
sleep 20
python ./taskgen.py --learning-rate 1e-1 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 16 --transformer-depth 2 4 10 --transformer-embed-dim 32 128 --regularization-l 0 3 --dataset-features featuresCVAEGCN_1_2000.h5 featuresCVAEGCN_2001_4000.h5 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer3-cvagen --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 2 &
wait

### FullRun - both datasets
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --sequence-name transformer3 --cuda-device-num 0 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 2  --transformer-depth 10 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --regularization-l 3 --sequence-name transformer3 --cuda-device-num 1 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --regularization-l 3 --sequence-name transformer3 --cuda-device-num 2 &
wait
--dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5


### FullRun - ELMD
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --sequence-name transformer3 --cuda-device-num 0 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 2  --transformer-depth 10 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --regularization-l 3 --sequence-name transformer3 --cuda-device-num 1 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 1000 --save-best --disable-early-exit --regularization-l 3 --sequence-name transformer3 --cuda-device-num 2 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 &
wait


#### ResNet hyperparam search
python ./taskgen.py --learning-rate 1e-3 1e-4 --batch-size 64 --model ResNet16 ResNet36 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
sleep 20
python ./taskgen.py --learning-rate 1e-2 1e-5 --batch-size 64 --model ResNet16 ResNet36 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 &
wait

#### ResNet hyperparam search + BatchNorm
python ./taskgen.py --learning-rate 1e-3 1e-4 1e-5 --batch-size 64 --model ResNet16BN ResNet36BN --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
wait

# Resnet deeper models, smaller lr
python ./taskgen.py --learning-rate 1e-7 1e-6 5e-5 --batch-size 64 --model ResNet54 ResNet36 ResNet100 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 &
wait

# Resnet deeper model
python ./taskgen.py --learning-rate 8e-5 6e-5 3e-5 1e-5 --batch-size 64 --model ResNet36 ResNet150 ResNet100 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 --regularization-l 3 --regularization-lambda 1e-4 1e-5 &
wait

# Transformer micro learning rate
python ./taskgen.py --learning-rate 1e-2 3e-2 5e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --disable-early-exit --regularization-l 3 --regularization-lambda 1e-4 1e-5 --sequence-name transformer3 --cuda-device-num 0 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --script emotion-gait.py &
sleep 20
python ./taskgen.py --learning-rate 1e-2 3e-2 5e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4  --transformer-depth 4 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --disable-early-exit --sequence-name transformer3 --cuda-device-num 1 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --script emotion-gait.py &
wait


# Resnet best + Affective features
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only --affective-features
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 --batch-size 64 --model ResNet36 ResNet100 --regularization-l 3 --regularization-lambda 1e-3 1e-5 --affective-features --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet-aff --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 --memmap &
sleep 20
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 --batch-size 64 --model ResNet36 ResNet100 ResNet150 --affective-features --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet-aff --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 --memmap &
sleep 20
python ./taskgen.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4 2  --transformer-depth 4 10 --transformer-embed-dim 32 --affective-features --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --regularization-l 3 --regularization-lambda 1e-3 1e-5 --sequence-name transformer3-aff --cuda-device-num 2 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --script emotion-gait.py --memmap &
sleep 20
python ./taskgen.py --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 4 2  --transformer-depth 4 10 --transformer-embed-dim 32 --affective-features --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --sequence-name transformer3-aff --cuda-device-num 2 --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --script emotion-gait.py --memmap &
wait

# Resnet best ELMD
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 --batch-size 64 --model ResNet36 ResNet100 --regularization-l 3 --regularization-lambda 1e-3 1e-5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 --memmap --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 &
sleep 20
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 --batch-size 64 --model ResNet36 ResNet100 ResNet150 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 4 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 --memmap --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 &
wait


# Transformer both datasets hyperparam search 
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
python ./taskgen.py --learning-rate 1e-1 1e-2 3e-2 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 2  --transformer-depth 2 4 10 --transformer-embed-dim 32 64 --regularization-l 0 3 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer4 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 --memmap &
sleep 20
python ./taskgen.py --learning-rate 1e-1 1e-2 3e-2 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 4 --transformer-depth 2 4 10 --transformer-embed-dim 32 64 --regularization-l 0 3 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer4 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 --memmap &
sleep 20
python ./taskgen.py --learning-rate 1e-1 1e-2 3e-2 5e-2 1e-3 --batch-size 64 --hidden-size 64 --transformer-heads 16 --transformer-depth 2 4 10 --transformer-embed-dim 32 64 --regularization-l 0 3 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name transformer4 --template template_hpc.sh --num-tasks-in-parallel 2 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 2 --memmap &
wait


# Resnet best 
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 1e-6 --batch-size 64 --model ResNet36 ResNet100 ResNet150 --regularization-l 3 --regularization-lambda 1e-3 1e-5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 0 --memmap &
sleep 20
python ./taskgen.py --learning-rate 8e-5 5e-5 1e-5 1e-6 --batch-size 64 --model ResNet36 ResNet100 ResNet150 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --script emotion-gait.py --sequence-name resnet --template template_hpc.sh --num-tasks-in-parallel 3 --num-cuda-devices-per-task 1 --epochs 5000 --save-best --cuda-device-num 1 --memmap &
wait


# Phased LSTM and SotaLSTM check on v100
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
python ./emotion-gait.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name v100speed --epochs 5000 --save-best --cuda-device-num 0 --memmap &
sleep 20
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --dataset-features features_ELMD.h5 --dataset-labels labels_ELMD.h5 --model SotaLSTM --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name v100speed --epochs 5000 --save-best --cuda-device-num 0 --memmap &
sleep 20
python ./emotion-gait.py --learning-rate 5e-3 --batch-size 128 --hidden-size 128 --model ModelRNN --layernorm --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name v100speed --epochs 5000 --save-best --cuda-device-num 0 --memmap &
wait

python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-5-run-21-07-24--10-21-27.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-12-run-21-07-24--11-04-31.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-18-run-21-07-24--11-42-12.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-24-run-21-07-24--12-07-41.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-30-run-21-07-24--12-40-38.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-35-run-21-07-24--13-02-12.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-42-run-21-07-24--13-30-07.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-48-run-21-07-24--14-03-01.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-54-run-21-07-24--14-54-47.sh
./results/transformer4-21-07-24-10-07-29/scripts/transformer4-60-run-21-07-24--15-50-18.sh


# Transformer best
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
python ./emotion-gait.py  --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 16  --transformer-depth 2 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --disable-early-exit --sequence-name transformer4 --cuda-device-num 0 --dataset-features features.h5 --dataset-labels labels.h5 --memmap &
sleep 20
python ./emotion-gait.py  --learning-rate 1e-2 --batch-size 64 --hidden-size 64 --transformer-heads 16  --transformer-depth 2 --transformer-embed-dim 32 --model Transformer2 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --disable-early-exit --sequence-name transformer4 --cuda-device-num 1 --memmap &
wait


# Resnet best
python ./emotion-gait.py --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --memmap-only
python ./emotion-gait.py --learning-rate 5e-5 --batch-size 64 --model ResNet36 --regularization-l 3 --regularization-lambda 1e-5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --cuda-device-num 0 --memmap --dataset-features features.h5 --dataset-labels labels.h5 &
sleep 20
python ./emotion-gait.py --learning-rate 5e-5 --batch-size 64 --model ResNet36 --regularization-l 3 --regularization-lambda 1e-5 --rotate-y --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --epochs 600 --save-best --cuda-device-num 0 --memmap &
wait

python ./emotion-gait.py --memmap-only
python ./emotion-gait.py --learning-rate 1e-2 --batch-size 64 --hidden-size 128 --model SotaLSTM --save-artefacts --use-cuda --epochs 600 --save-artefacts --save-best --sequence-name params4 --cuda-device-num 0 &
sleep 20
python ./emotion-gait.py --learning-rate 1e-3 --batch-size 64 --hidden-size 256 --model Model-P-LSTM --p-lstm-alpha 1e-5 --p-lstm-r-on 0.1 --save-artefacts --use-cuda --sequence-name params4 --epochs 600 --save-best --cuda-device-num 1 &
wait

