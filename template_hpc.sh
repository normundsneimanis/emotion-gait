#!/bin/sh
module load conda
eval "$(conda shell.bash hook)"
source activate normunds_env
export LD_LIBRARY_PATH=~/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH
mkdir -p /scratch/nneimanis
mkdir -p /scratch/nneimanis/tmp
mkdir -p /scratch/nneimanis/data
export TMPDIR=/scratch/nneimanis/tmp
export TEMP=/scratch/nneimanis/tmp
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

if [[ $EUID -eq 0 ]]; then
	ulimit -n `cat /proc/sys/fs/file-max`
fi

cd /mnt/home/abstrac01/normunds_neimanis
