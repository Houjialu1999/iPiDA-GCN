import argparse
import json

import numpy as np
import os
import pandas as pd
import torch as t
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from model import GCMC
from torch import nn, optim
from sklearn.preprocessing import PolynomialFeatures

from utils import rwr


# set random generator seed to allow reproducibility
t.manual_seed(12345)
np.random.seed(12345)

# assign cuda device ID
device = "cuda:0"
device = t.device('cuda')

def import_data():
    file_path = os.path.dirname(__file__)
    file_name = os.path.abspath(os.path.join(file_path, '../model/dataset/'))
    PS_seq = np.load(file_name + '/piSim.npy')
    DS_doid = np.load(file_name + '/DiseaSim.npy')
    PD_fold_label = np.load(file_name + '/PD_fold5_balance.npy')
    adjPD=np.load(file_name + '/adjPD.npy')
    PD_ben_ind_label=np.load(file_name + '/PD_ben_ind_label.npy')
    return PS_seq,DS_doid,PD_fold_label,adjPD,PD_ben_ind_label

def tst_metric(label, input, idx):
    """
    Monitor the performance on test data set.
    :param label: ground-truth, i.e. completed protein-HPO matrix
    :param input: predicted score, i.e. predicted protein-HPO matrix
    :param idx: identify which entries are in the test set
    :return: AUC and AUPR on test set
    """
    score = input.detach().cpu().numpy()[idx]
    return roc_auc_score(label, score), average_precision_score(label, score)


def fit(model, train_data, optimizer):
    """
    Predict full protein-HPO association matrix.
    :param model: instance of model
    :param train_data: assembled training data
    :param optimizer: instance of optimizer
    :return: predicted scores, i.e. predicted protein-HPO association matrix
    """
    # turn to training mode
    model.train()
    # use MSE as the loss function
    criterion = nn.MSELoss(reduction='sum')
    # let learning_rate decrease by 50% at 500, 1000 and 2000-th epoch
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,range(300,3000,300), gamma=0.5)

    def train_epoch(i):
        """
        Conduct i-th training iteration
        :param i: which iteration is going on
        :return: loss on training set, AUC & AUPR on test set
        """
        model.zero_grad()

        score = model(train_data["pi_feature"],
                      train_data["dis_feature"],
                      train_data["pi_sim"],
                      train_data["dis_sim"],
                      train_data["relation"])
        trn_loss = criterion(train_data["train_annotation"], score)
        # print log info every 25 iterations
        if i % 25 == 0:
            tst_auc, tst_aupr = tst_metric(train_data["test_label"], score, train_data["test_idx"])
        else:
            tst_auc, tst_aupr = 0, 0
        trn_loss.backward()
        optimizer.step()
        scheduler.step()
        return trn_loss, tst_auc, tst_aupr

    # conduct training for total of 3000 iterations
    for epoch in range(3000):
        trn_loss, tst_auc, tst_aupr = train_epoch(epoch)
        print("Epoch", epoch, "\t", trn_loss.item(), "\t", tst_auc, "\t", tst_aupr)

    # return the final predicted score
    return model(train_data["pi_feature"],
                 train_data["dis_feature"],
                 train_data["pi_sim"],
                 train_data["dis_sim"],
                 train_data["relation"])

if __name__ == "__main__":
    PS_seq, DS_doid, PD_fold_label,adjPD,PD_ben_ind_label = import_data()
    # the number of piRNA and disease
    m_pi, n_dis = adjPD.shape
    pred_score=np.zeros((m_pi, n_dis))

    # extract protein features by Random Walk with Restart
    pi_features = rwr(PS_seq, 0.9)

    dis_features1 = rwr(DS_doid, 0.9)
    poly_dis = PolynomialFeatures(3, include_bias=False)
    dis_features = poly_dis.fit_transform(dis_features1)

    # load index
    train_index = np.where((PD_fold_label != 0))
    test_index = np.where((PD_fold_label == 0))

    train_mask=np.zeros((adjPD.shape))
    train_annotation = np.zeros((adjPD.shape))
    test_annotation = np.zeros((adjPD.shape))

    # apply mask to extract known annotations
    train_mask[train_index]=1

    # construct protein-phenotype block matrix, size: (m + n, m + n)
    rel = np.concatenate((
        np.concatenate((np.zeros((m_pi, m_pi)), train_annotation), axis=1),
        np.concatenate((train_annotation.T, np.zeros((n_dis, n_dis))), axis=1)
    ), axis=0)

    # emphasize the positive labels
    train_annotation[train_mask == 1] = 5

    # assemble the training data
    train_data = {
        "pi_feature": t.FloatTensor(pi_features).to(device),
        "dis_feature": t.FloatTensor(dis_features).to(device),
        "pi_sim": t.FloatTensor(PS_seq).to(device),
        "dis_sim": t.FloatTensor(DS_doid).to(device),
        "train_annotation": t.FloatTensor(train_annotation).to(device),
        "relation": t.FloatTensor(rel).to(device),
        "test_annotation": t.FloatTensor(test_annotation[test_index]).to(device),
        "test_label": adjPD[test_index],
        "test_idx": test_index
    }

    # create our model
    model = GCMC(m_pi, n_dis,pi_features,dis_features)
    model.to(device)
    # create optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1.)  #调节learning rate

    # make prediction
    Y_pred = fit(model, train_data, optimizer)
    Y_pred = Y_pred.detach().cpu().numpy()

    # retain only predictions not in train set
    pred_score[test_index]=Y_pred[test_index]



