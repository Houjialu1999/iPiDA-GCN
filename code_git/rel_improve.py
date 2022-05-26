import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def rel_RF(PD_ben_ind_label,PD_fold_label,pi_feature,dis_feature):

    # index of train samples and test samples
    train_index = np.where(PD_fold_label !=0)
    test_index = np.where(PD_ben_ind_label!=0)

    # construct train label and test label
    PD_ben = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind = np.zeros(PD_ben_ind_label.shape, dtype=int)
    PD_ind[PD_ben_ind_label == -1] = 1
    PD_ben[PD_ben_ind_label == 1] = 1
    train_label = PD_ben[train_index]
    test_label = PD_ind[test_index]

    # constuct data feature for training set
    train_data = []
    for (cur_train_row, cur_train_col) in zip(train_index[0], train_index[1]):
        cur_feature = pi_feature[cur_train_row].tolist()
        cur_feature.extend(dis_feature[cur_train_col].tolist())
        train_data.append(cur_feature)

    # construct data feature for test set
    test_data = []
    for (cur_test_row, cur_test_col) in zip(test_index[0], test_index[1]):
        cur_feature = pi_feature[cur_test_row].tolist()
        cur_feature.extend(dis_feature[cur_test_col].tolist())
        test_data.append(cur_feature)

    #train GBDT model and make prediction
    GBDT_model = GradientBoostingClassifier()
    GBDT_model.fit(train_data, train_label)

    Score_ind_GBDT = GBDT_model.predict(test_data)
    Score_GBDT=np.zeros((PD_fold_label.shape))
    Score_GBDT[test_index]=Score_ind_GBDT

    pos_index=np.where((PD_ben_ind_label==1))
    adj=np.zeros((PD_ben_ind_label.shape))
    adj[pos_index]=1
    rel_RF=np.maximum(Score_GBDT,adj)
    
    return rel_RF

if __name__ == '__main__':
    # load data
    file_path = os.path.dirname(__file__)
    file_name = os.path.abspath(os.path.join(file_path, '../dataset/'))
    PS_seq = np.load(file_name + '/piSim.npy')
    DS_doid = np.load(file_name + '/DiseaSim.npy')
    PD_fold_label = np.load(file_name + '/PD_fold5_balance_col4.npy')
    adjPD=np.load(file_name + '/adjPD.npy')
    PD_ben_ind_label=np.load(file_name + '/PD_ben_ind_label_balance_col4.npy')

    #set diagonal elements 1
    row_id, col_id = np.diag_indices_from(PS_seq)
    PS_seq[row_id, col_id] = 1

    row_id, col_id = np.diag_indices_from(DS_doid)
    DS_doid[row_id, col_id] = 1

    # 构造当前折的benchmark和independent的训练集和测试集
    rel_RF = rel_RF(PD_ben_ind_label,PD_fold_label,PS_seq,DS_doid)
    np.save(file_name+'/rel_GBDT.npy',rel_RF)