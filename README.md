# Human gait emotion recognition deep learning research project
Based on [STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits](https://arxiv.org/abs/1910.12906v1)

## Dataset

#### Get dataset
Download original dataset  <https://go.umd.edu/emotion-gait>, [mirror](http://nneimanis.id.lv/emotion-gait/)

#### Reading, viewing dataset
Dataset files containing gaits are
- emotion-gait/features.h5 Gaits collected in [STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits](https://arxiv.org/abs/1910.12906v1)
- emotion-gait/features_ELMD.h5 ELMD Gaits taken from [Take an Emotion Walk: Perceiving Emotions from Gaits Using Hierarchical Attention Pooling and Affective Mapping](http://arxiv.org/abs/1911.08708v2)
- emotion-gait/featuresCVAEGCN_1_2000.h5 Generated gaits no. 0-2000
- emotion-gait/featuresCVAEGCN_2001_4000.h5 Generated gaits no. 2000-4000


View random dataset can be done choosing one of dataset files

	python readEmotionGaitDS.py -f .\emotion-gait\features.h5 -l .\emotion-gait\labels.h5
	python readEmotionGaitDS.py -f .\emotion-gait\features_ELMD.h5 -l .\emotion-gait\labels_ELMD.h5
	python readEmotionGaitDS.py -f .\emotion-gait\featuresCVAEGCN_1_2000.h5
	python readEmotionGaitDS.py -f .\emotion-gait\featuresCVAEGCN_2001_4000.h5

Specific gait can be selected with dataset id

	--view-gait <num>

View animated dataset [online](http://nneimanis.id.lv/emotion-gait/index.php)

Dataset is available for download for training online [online](http://nneimanis.id.lv/emotion-gait/emotion-gait.h5)

## Running training script

    python ./emotion-gait.py --learning-rate 1e-2 --batch-size 128 --hidden-size 256 --model SotaLSTM --remove-zeromove --center-root --normalize-gait-sizes --scale --save-artefacts --use-cuda --sequence-name params2 --epochs 5000 --save-best --overfit-exit-percent 30


#### Dependencies

Set up conda environment

    conda create -n conda_env
    conda activate conda_env
    conda install -c anaconda mysql-connector-python
    conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
    conda install h5py tqdm
    conda install -c conda-forge python-dotenv
    conda install -c anaconda requests

    conda install -c conda-forge matplotlib
    conda install -c anaconda mysql-python

    conda install -c conda-forge scikit-learn 
    conda install -c conda-forge tensorboardx
    conda install -c conda-forge tensorboard
    conda install -c conda-forge tensorflow
    conda install -c conda-forge einops
    pip3 install thop
    pip3 install python_papi
    sudo apt-get install papi-tools

#### HPC
Activate conda environment

	conda activate conda_env

Show queue

	showq -r
	qstat -r
	
Submit task

	qsub ./start.sh

View task status

	qstat
	
Delete task

	qdel <task_id>
	
Check job

	checkjob -vvv <task_id>

Connect to node running task (from `showq -r`)

	ssh wn57
	
View resource utilization

	htop 
	dstat 
	nvidia-smi 
	iostat  
	nfsstat
	

View Tensorboard results

	tensorboard.exe --logdir=C:\workspace\emotion-gait\artefacts\seq_default\


