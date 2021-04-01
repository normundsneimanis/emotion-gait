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
