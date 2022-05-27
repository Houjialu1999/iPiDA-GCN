# iPiDA-GCN
iPiDA-GCN: identification of piRNA-disease associations based on Graph Convolutional Network

# Dependencies
* python3.6 with pytorch 1.9.0 and torch-geometric 2.0.3
* Nvidia GPU with CUDA 11.1
* numpy 1.19.5
* sklearn 0.24.2

# Data description
* piRNA_information.xlsx：list of piRNA names and piRNAs' corresponding number
* disease_information.xlsx: list of disease names and diseases' corresponding number
* adjPD.npy: piRNA-disease association matrix, a(i,j) represents the association between i-th piRNA and j-th disease
* piSim.npy: piRNA-piRNA similarity matrix
* DiseaSim.npy: disease-disease similarity matrix
* PD_ben_ind_label.npy: benchmark dataset and independent dataset division, where the label 1 and -20 represent positive and negtive samples in benchmark dataset, label -1 and -10 represent positive and negtive samples in independent dataset

# Run step
1. run `rel_improve.py` for piRNA0-disease association predetection
2. run `train_PD.py` to train the model and obtain the predicted scores for piRNA-disease associations
3. run `evaluate.py` to evaluate the model performance with AUC and AUPR

