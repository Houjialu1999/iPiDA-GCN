import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc, precision_recall_curve,f1_score

class evaluation_index:
    def __init__(self, label, score, ndcg_k):
        self.score = score
        self.label = label
        self.ndcg_k = ndcg_k

    def evaluation(self):
        # AUC
        self.AUC = roc_auc_score(self.label, self.score)

        # AUPR
        Cur_precision, Cur_recall, _thresholds = precision_recall_curve(self.label, self.score)
        self.AUPR = auc(Cur_recall, Cur_precision)
        # NDCG@k
        df = pd.DataFrame(columns=["y_pred", "y_true"], data=np.array([list(self.score), list(self.label)]).T)
        self.ndcg = self.get_ndcg(df, self.ndcg_k)
        # average precision
        self.ap = average_precision_score(self.label, self.score)
        # Reciprocal Rank
        self.rr = self.reciprocal_rank(self.label, self.score)
        # Reciprocal Rank@10
        self.rr10 = self.reciprocal_rank10(self.label, self.score)
        # Hit Ratio
        self.HR = self.hit_ratio(self.label, self.score, 10)
        # ROC
        self.ROC = []
        temp = list(range(1, 22, 1))
        # for i in range(1, 51, 1):
        for i in temp:
            self.ROC.append(self.ROC_x(self.label, self.score, i))

    # ---------------------------------------------------------------------------------------
    # NDCG@k
    def get_dcg(self, y_pred, y_true, k):
        df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
        df = df.sort_values(by="y_pred", ascending=False)
        df = df.iloc[0:k, :]
        dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count() + 1) + 1)
        dcg = np.sum(dcg)
        return dcg

    def get_ndcg(self, df, k):
        dcg = self.get_dcg(df["y_pred"], df["y_true"], k)
        idcg = self.get_dcg(df["y_true"], df["y_true"], k)
        ndcg = dcg / idcg
        return ndcg

    # ---------------------------------------------------------------------------------------
    # reciprocal rank
    def reciprocal_rank(self, label, score):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        pos_loc = np.where(descend_label == 1)

        rr = 1/(pos_loc[0][0]+1)
        return rr

    # ---------------------------------------------------------------------------------------
    # reciprocal rank @10
    def reciprocal_rank10(self, label, score):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        gt = [1]
        score = 0.0
        for rank, item in enumerate(descend_label[:10]):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break

        return score
    # ---------------------------------------------------------------------------------------
    # reciprocal rank @10
    def hit_ratio(self, label, score, x):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        if x > descend_label.size:
            x = descend_label.size

        pos_num = 0
        for i in range(x):
            if descend_label[i] == 1:
                pos_num += 1

        return pos_num

    # ---------------------------------------------------------------------------------------
    # ROCx
    def ROC_x(self, label, score, x):
        if x > np.where(label == 0)[0].size:
            x = np.where(label == 0)[0].size

        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        roc_value = self.cal_roc_mu_at(x, descend_label)
        return roc_value

    def cal_roc_mu_at(self, level, y_true):
        # 统计所有的true positive
        tp = np.count_nonzero(y_true)
        # print(tp)
        # 统计level个false positive之前的tp之和
        all_fp = np.count_nonzero(~y_true.astype(bool))
        if all_fp < level:
            # 如果不足就填充False，因为，没检索出来
            fp = all_fp
            yt_level = y_true
        else:
            df = pd.DataFrame(y_true.astype(bool), columns=["label"])
            fp_index = df.loc[~df.loc[:, "label"]].index[level - 1]
            yt_level = df.loc[:fp_index].to_numpy(int).reshape((-1,))
        fp = np.count_nonzero(~yt_level.astype(bool))
        # print(fp)

        # print(yt_level)
        cumsum_yt_level = np.cumsum(yt_level)
        area = cumsum_yt_level[~yt_level.astype(bool)].sum()
        # print(area)
        if tp != 0 and fp != 0:
            roc_at_score = area / (tp * fp)
        elif tp != 0 and fp == 0:
            roc_at_score = 1.0
        elif tp == 0 and fp != 0:
            roc_at_score = 0.0
        return roc_at_score

def evaluation_ind1(PD, score, label_matrix): #按列计算AUC等
    sample_index = np.where(np.sum(PD, axis=1) > 0)[0]
    MD = PD[sample_index]
    label_matirx = label_matrix[sample_index]
    Score = score[sample_index]

    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    for iter_row in sample_index:
        print(iter_row)
        Pos_loc = np.where(label_matirx[iter_row] == -1)
        Neg_loc = np.where(label_matirx[iter_row] == -2)
        sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]
        y_label = MD[iter_row][sample_loc]
        y_scores = Score[iter_row][sample_loc]

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 791)
        cur_miRNA_performence.evaluation()

        AUC_all.append(cur_miRNA_performence.AUC)
        AUPR_all.append(cur_miRNA_performence.AUPR)
        NDCG10_all.append(cur_miRNA_performence.ndcg)
        AP_all.append(cur_miRNA_performence.ap)
        RR_all.append(cur_miRNA_performence.rr)
        HR_all.append(cur_miRNA_performence.HR)
        RR10_all.append(cur_miRNA_performence.rr10)
        ROC_all.append(cur_miRNA_performence.ROC)


    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(np.where(label_matirx < 0)[0].size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC

def evaluation_ind2(PD, score, label_matrix):
    test_index = np.where(label_matrix < 0)
    test_label = PD[test_index]
    test_score = score[test_index]

    performence = evaluation_index(test_label, test_score, 10)
    performence.evaluation()

    return performence.AUC, performence.AUPR, performence.ndcg, performence.ap, performence.rr, performence.rr10, performence.HR, performence.ROC

def evaluation_ind3(PD, score, label_matrix):#按piRNA来计算
    sample_index = np.where(np.sum(PD, axis=1) > 0)[0]
    MD = PD[sample_index]
    label_matirx = label_matrix[sample_index]
    Score = score[sample_index]

    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    for iter_row in sample_index:
        # print(iter_row)
        Pos_loc = np.where(label_matirx[iter_row] == -1)
        Neg_loc = np.where(label_matirx[iter_row] == -2)
        if Neg_loc[0].size != 0:
            sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]  #？
            y_label = MD[iter_row][sample_loc]
            y_scores = Score[iter_row][sample_loc]

            cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
            cur_miRNA_performence.evaluation()

            AUC_all.append(cur_miRNA_performence.AUC)
            AUPR_all.append(cur_miRNA_performence.AUPR)
            NDCG10_all.append(cur_miRNA_performence.ndcg)
            AP_all.append(cur_miRNA_performence.ap)
            RR_all.append(cur_miRNA_performence.rr)
            HR_all.append(cur_miRNA_performence.HR)
            RR10_all.append(cur_miRNA_performence.rr10)
            ROC_all.append(cur_miRNA_performence.ROC)


    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(np.where(label_matirx < 0)[0].size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC


def evaluation_Data2_ben(MD, MD_none_FP, Score):

    FP_id_matrix = MD_none_FP - MD
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    for iter_row in range(MD.shape[0]):
        cur_remover_FP_loc = np.where(FP_id_matrix[iter_row] == 0)

        y_label = MD[iter_row]
        y_label = y_label[cur_remover_FP_loc]
        y_scores = Score[iter_row]
        y_scores = y_scores[cur_remover_FP_loc]

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
        cur_miRNA_performence.evaluation()

        AUC_all.append(cur_miRNA_performence.AUC)
        AUPR_all.append(cur_miRNA_performence.AUPR)
        NDCG10_all.append(cur_miRNA_performence.ndcg)
        AP_all.append(cur_miRNA_performence.ap)
        RR_all.append(cur_miRNA_performence.rr)
        HR_all.append(cur_miRNA_performence.HR)
        RR10_all.append(cur_miRNA_performence.rr10)
        ROC_all.append(cur_miRNA_performence.ROC)


    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(Score.size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)


    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC

def ben_new_associations_combine(Score, PD_fold1): #按piRNA计算AUC等
    PD_fold = np.copy(PD_fold1)
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    MD = np.zeros(PD_fold.shape, dtype=int)
    MD[PD_fold > 0] = 1

    for iter_row in range(PD_fold.shape[0]):

        Pos_loc = np.where(PD_fold[iter_row] > 0)
        Neg_loc = np.where(PD_fold[iter_row] < 0)
        if (Pos_loc[0].size != 0) & (Neg_loc[0].size != 0):
            # print(iter_row)
            sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]
            y_label = MD[iter_row][sample_loc]
            y_scores = Score[iter_row][sample_loc]

            cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
            cur_miRNA_performence.evaluation()

            AUC_all.append(cur_miRNA_performence.AUC)
            AUPR_all.append(cur_miRNA_performence.AUPR)
            NDCG10_all.append(cur_miRNA_performence.ndcg)
            AP_all.append(cur_miRNA_performence.ap)
            RR_all.append(cur_miRNA_performence.rr)
            HR_all.append(cur_miRNA_performence.HR)
            RR10_all.append(cur_miRNA_performence.rr10)
            ROC_all.append(cur_miRNA_performence.ROC)

    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(np.where( PD_fold != 0)[0].size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC

def ben_new_associations_fold(Score, PD_fold1, fold_num): #对每一折进行测试
    PD_fold = np.copy(PD_fold1)
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    MD = np.zeros(PD_fold.shape, dtype=int)
    MD[PD_fold == (fold_num+1)] = 1

    for iter_row in range(PD_fold.shape[0]):

        Pos_loc = np.where(PD_fold[iter_row] == (fold_num+1))
        Neg_loc = np.where(PD_fold[iter_row] == -(fold_num+1))
        if (Pos_loc[0].size != 0) & (Neg_loc[0].size != 0):
            # print(iter_row)
            sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]
            y_label = MD[iter_row][sample_loc]
            y_scores = Score[iter_row][sample_loc]

            cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
            cur_miRNA_performence.evaluation()

            AUC_all.append(cur_miRNA_performence.AUC)
            AUPR_all.append(cur_miRNA_performence.AUPR)
            NDCG10_all.append(cur_miRNA_performence.ndcg)
            AP_all.append(cur_miRNA_performence.ap)
            RR_all.append(cur_miRNA_performence.rr)
            HR_all.append(cur_miRNA_performence.HR)
            RR10_all.append(cur_miRNA_performence.rr10)
            ROC_all.append(cur_miRNA_performence.ROC)

    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(np.where( PD_fold != 0)[0].size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC



def evaluation_Data2_ben_FP(MD, MD_none_FP, Score):

    label_matrix = np.zeros(MD.shape)
    label_matrix = MD_none_FP - MD
    label_matrix[MD_none_FP == 0] = 1

    FP_id_matrix = MD_none_FP - MD
    AUC_all = []
    AUPR_all = []
    NDCG10_all = []
    AP_all = []
    RR_all = []
    RR10_all = []
    ROC_all = []
    HR_all = []
    for iter_row in range(MD.shape[0]):
        cur_remover_FP_loc = np.where(label_matrix[iter_row] == 1)

        y_label = MD_none_FP[iter_row]
        y_label = y_label[cur_remover_FP_loc]
        y_scores = Score[iter_row]
        y_scores = y_scores[cur_remover_FP_loc]
        if y_label.sum() == 0:
            continue

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
        cur_miRNA_performence.evaluation()

        AUC_all.append(cur_miRNA_performence.AUC)
        AUPR_all.append(cur_miRNA_performence.AUPR)
        NDCG10_all.append(cur_miRNA_performence.ndcg)
        AP_all.append(cur_miRNA_performence.ap)
        RR_all.append(cur_miRNA_performence.rr)
        HR_all.append(cur_miRNA_performence.HR)
        RR10_all.append(cur_miRNA_performence.rr10)
        ROC_all.append(cur_miRNA_performence.ROC)


    AUC = round(np.mean(AUC_all), 4)
    AUPR = round(np.mean(AUPR_all), 4)
    NDCG10 = round(np.mean(NDCG10_all), 4)
    MAP = round(np.mean(AP_all), 4)
    MRR = round(np.mean(RR_all), 4)
    MRR10 = round(np.mean(RR10_all), 4)
    HR = round(np.sum(HR_all)/(Score.size), 4)
    ROC = np.round(np.mean(ROC_all, axis=0), decimals=4)


    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, HR, ROC

def evaluation_Data2_ben_ROCx_all(MD, MD_none_FP, Score, x):
    FP_id_matrix = MD_none_FP - MD
    ROCx_all = []
    for iter_row in range(MD.shape[0]):
        cur_remover_FP_loc = np.where(FP_id_matrix[iter_row] == 0)

        y_label = MD[iter_row]
        y_label = y_label[cur_remover_FP_loc]
        y_scores = Score[iter_row]
        y_scores = y_scores[cur_remover_FP_loc]

        cur_remover_FP_loc = np.where(FP_id_matrix[iter_row] == 0)

        y_label = MD[iter_row]
        y_label = y_label[cur_remover_FP_loc]
        y_scores = Score[iter_row]
        y_scores = y_scores[cur_remover_FP_loc]

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
        ROCx_all.append(cur_miRNA_performence.ROC_x(y_label, y_scores, x))

    return ROCx_all

def evaluation_Data2_ind_ROCx_all(MD, score, label_matrix, x):
    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    MD = MD[sample_index]
    label_matirx = label_matrix[sample_index]
    Score = score[sample_index]

    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    ROCx_all = []
    for iter_row in sample_index:
        Pos_loc = np.where(label_matirx[iter_row] == -1)
        Neg_loc = np.where(label_matirx[iter_row] == -2)
        sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]
        y_label = MD[iter_row][sample_loc]
        y_scores = Score[iter_row][sample_loc]

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
        ROCx_all.append(cur_miRNA_performence.ROC_x(y_label, y_scores, x))

    return ROCx_all

def evaluation_Data3_ind_ROC1_100(MD, Score):

    ROCx_all = []
    for x in range(5, 105, 5):
        cur_x_Result = []
        for iter_row in range(MD.shape[0]):
            y_label = MD[iter_row]
            y_scores = Score[iter_row]

            cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
            cur_x_Result.append(cur_miRNA_performence.ROC_x(y_label, y_scores, x))
        ROCx_all.append(cur_x_Result)

    return ROCx_all

def evaluation_Data3_ind_ROCx_all(MD, Score, x):

    ROCx_all = []

    for iter_row in range(MD.shape[0]):
        y_label = MD[iter_row]
        y_scores = Score[iter_row]

        cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
        ROCx_all.append(cur_miRNA_performence.ROC_x(y_label, y_scores, x))

    return ROCx_all

def evaluation_Data3_ind_AUC_all(MD, Score):
    ROCx_all = []

    for iter_row in range(MD.shape[0]):
        y_label = MD[iter_row]
        y_scores = Score[iter_row]

        ROCx_all.append(roc_auc_score(y_label, y_scores))

    return ROCx_all

def evaluation_Data2_ben_AUC_all(MD, MD_none_FP, Score):
    FP_id_matrix = MD_none_FP - MD
    AUC_all = []
    for iter_row in range(MD.shape[0]):
        cur_remover_FP_loc = np.where(FP_id_matrix[iter_row] == 0)

        y_label = MD[iter_row]
        y_label = y_label[cur_remover_FP_loc]
        y_scores = Score[iter_row]
        y_scores = y_scores[cur_remover_FP_loc]

        AUC_all.append(roc_auc_score(y_label,  y_scores))

    return AUC_all

def evaluation_Data2_ind_ROC5_100(MD, score, label_matrix):
    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    MD = MD[sample_index]
    label_matirx = label_matrix[sample_index]
    Score = score[sample_index]

    sample_index = np.where(np.sum(MD, axis=1) > 0)[0]
    ROCx_all = []
    for x in range(5, 105, 5):
        cur_x_Result = []
        for iter_row in sample_index:
            Pos_loc = np.where(label_matirx[iter_row] == -1)
            Neg_loc = np.where(label_matirx[iter_row] == -2)
            sample_loc = np.concatenate((Pos_loc, Neg_loc), axis=1)[0]
            y_label = MD[iter_row][sample_loc]
            y_scores = Score[iter_row][sample_loc]

            cur_miRNA_performence = evaluation_index(y_label, y_scores, 10)
            cur_x_Result.append(cur_miRNA_performence.ROC_x(y_label, y_scores, x))
        ROCx_all.append(cur_x_Result)

    return ROCx_all

def evaluation_all(score,label):
    performence=evaluation_index(label,score,10)
    performence.evaluation()

    AUC = round(performence.AUC, 4)
    AUPR = round(performence.AUPR, 4)
    NDCG10 = round(performence.ndcg, 4)
    MAP = round(performence.ap, 4)
    MRR = round(performence.rr, 4)
    MRR10 = round(performence.rr10, 4)
    #HR = round(np.sum(HR_all) / (np.where(PD_fold != 0)[0].size), 4)
    ROC = np.round(performence.ROC, decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, ROC


