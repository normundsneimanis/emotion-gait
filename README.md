# Emotion Gait AI research project
Based on [STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits](https://arxiv.org/abs/1910.12906v1)

## Dataset

#### Get dataset
Download original dataset  <https://go.umd.edu/emotion-gait>, [mirror](http://nneimanis.id.lv/emotion-gait/emotion-gait.zip)

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

## Runing training script
Training script emotion-gait.py can be executed on [Google Colab](https://colab.research.google.com/drive/1QuZP5JA2TmSBc-JgAXzJ45_xjWFiNLs-?usp=sharing)


#### Dependencies

	pip.exe install tensorflow
	pip.exe install tensorboardX
	pip.exe install pywin32
	pip.exe install future
	pip.exe install moviepy
	conda install ffmpeg
	conda install h5py

#### HPC
Set conda environment

	conda activate conda_env

Show queue

	showq -r
	
	qstat -r
	
Submit task

	qsub -N normunds_neimanis ./start.sh

View task status

	qstat
	
Delete task

	qdel <task_id>
	
Check job

	checkjob -vvv <task_id>

Connect to node runnign task (from `showq -r`)

	ssh wn57
	
View resource utilization

	htop 
	dstat 
	nvidia-smi 
	iostat  
	nfsstat
	

Conda environment creation commands

	conda create -n normunds_env python=<version>
	conda install pytorch torchvision cudatoolkit h5py tqdm tensorflow
	conda env remove -n ENV_NAME
	conda create --clone py35 --name py35-2


View Tensorboard results

	tensorboard.exe --logdir=C:\workspace\emotion-gait\artefacts\seq_default\


