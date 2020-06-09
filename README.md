# EEG_Stream_3D

# Spato-Temporal Representation of Electoencephalogram for Emotion Recognition using Three-Dimensional Convolution Neural Network

## 1. Requirements

see the requirements.txt file for install python module

## 2. Data preprataion (stream make)

### 2.1 Download DEAP dataset
download DEAP dataset in /stream_make/data_preprocessed_python

### 2.2 Prepare EEG stream dataset (for traning)
after unzip the files run :
`python seq_make_1sec_subj_1.py`

## 3. EEG stream traning

run : `python train.py ./config/some_configuration_file.cfg`

For training C3D network, use train_c3d.cfg

For traning R2+1D network, use train_r2plus1d_64.cfg
