import numpy as np
import os
from evaluation import evaluation_fun


file_path = os.path.dirname(__file__)
file_name = os.path.abspath(os.path.join(file_path, '../dataset/'))
adjPD=np.load(file_name + '/adjPD.npy')
Y_pred=np.load(file_name + '/predicted_score.npy')

# retain only predictions not in train set
cur_AUC, cur_AUPR, cur_NDCG10, cur_MAP, cur_MRR, cur_MRR10, cur_ROC = \
		evaluation_fun.evaluation_all(Y_pred[test_index], adjPD[test_index])
print('AUC:',cur_AUC,'........','AUPR:',cur_AUPR)

