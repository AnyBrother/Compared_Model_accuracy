# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/4 10:40
@Software: PyCharm
@Author  : ykp
@Version : 1.0 实现从excel(第一行为列名,最后一列是标签列, 且excel文件与该python代码处于同一个文件夹内)
                读取数据, 计算指定模型的精度,并将精度结果输出成excel文件.
               Enable reading data from 'excel' data document (First row is 'name of index',
                last column is 'labels 0/1'), which is located in the same file with this
                python code, auto-calculate accuracy of desired models and output results
                into 'excel'.
            1.1 新增几个二分类的精度标准.(Added: Brier_score, Kappa, Inverse_precision, Jaccard,
                                        Youden’s Index, Gini coefficient)
                Added New Accuracy measurements for binary classification.
"""
import os
import pandas as pd
import numpy as np


from sklearn.metrics import roc_curve, auc, confusion_matrix  # 导入 获取精度判别的计算包
from sklearn import svm  # 导入 SVM 模型
from sklearn.linear_model import LogisticRegression  # 导入 逻辑回归 模型
from sklearn.naive_bayes import GaussianNB  # 导入 高斯-贝叶斯 模型
from sklearn.tree import DecisionTreeClassifier  # 导入 决策树 模型
from sklearn.neighbors import KNeighborsClassifier  # 导入 KNN聚类 模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # 导入 随机森林+梯度boost集成 模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 导入 线性判别 模型
from sklearn.neural_network import MLPClassifier  # 导入 神经网络-多层感知器分类器 模型
from sklearn.ensemble import AdaBoostClassifier  # 导入 AdaBoost集成 模型

import time
import datetime
import pickle


class AccuracyMeasureYkp:
    # 计算精度结果
    """
    From Paper:
        Title: A novel ensemble method for credit scoring: Adaption of different imbalance ratios
        Authors: Hongliang He, Wenyu Zhang∗, Shuai Zhang
        Journal: Expert Systems With Applications
        Year&Volume: 98 (2018) 105–117
        DOI: 10.1016/j.eswa.2018.01.012
    """

    def __init__(self, y_real, y_pre_prob, y_pre_label):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_prob   —— 预测标签概率值
            y_pre_label  —— 预测得到的标签值
        """
        self.y_real = y_real
        self.y_pre_prob = y_pre_prob
        self.y_pre_label = y_pre_label

    def confusion_matrix_ykp(self):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_label  —— 预测得到的标签值
        Output:
            TP,TN,FP,FN  —— 混淆矩阵结果
        Confusion Matrix:
        =============================================================
                     预测
                  |  1                            0                             合计
          实 |  1 |  True  Positive(TP)           False Negative(FN)            Actual Positive=(TP+FN)
          际 |  0 |  False Positive(FP)           True  Negative(TN)            Actual Negative=(FP+TN)
          合   计 |  Predicted Positive(TP+FP)    Predicted Negative(FN+TN)     TP+FP+FN+TN
        =============================================================
        """
        TN, FP, FN, TP = confusion_matrix(self.y_real, self.y_pre_label, labels=[0, 1]).ravel()
        return TN, FP, FN, TP

    def accuracy_ykp(self):
        """
        Input:
            TP,TN,FP,FN  —— 混淆矩阵结果
        Output:
            Accuracy     ——  One of the prevailing evaluation measure
                        and defined as the correct prediction sample size
                        divided by the total testing sample size.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Accuracy = (TP + TN) / (TP + FN + FP + TN)
        return Accuracy

    def f_measure_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Precision     —— 预测的1中判对的比例
            Recall        —— 真实的1中判对的比例
            F_measure     —— 又称为F1分数或F分数,是权衡Precision和Recall是使用精度和召回率的方法组合到一个度量上
            Type_II_error —— 第二类错误(将违约样本错判为非违约的比例)
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)  # Type-II error=1-Recall
        F_measure = 2 * Precision * Recall / (Precision + Recall)
        Type_II_error = 1 - Recall  # 第二类错误(将违约样本错判为非违约的比例)
        return Precision, Recall, F_measure, Type_II_error

    def g_means_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Sensitivity   ——  真实的1中判对的比例=True Positive Rate (TPR) or Recall
            Specificity   —— 真实的0中判对的比例
            G_Means     —— The higher G-Mean shows the balance between classes is reasonable
                        and has good performance in the binary classification model.
            Type_I_error —— 第一类错误(将非违约样本错判为违约的比例)
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        G_Means = np.sqrt(Sensitivity * Specificity)
        Type_I_error = 1 - Specificity  # 第一类错误(将非违约样本错判为违约的比例)
        return Sensitivity, Specificity, G_Means, Type_I_error

    def logistic_loss_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            Logistic_Loss ——  also known as log loss or cross-entropy loss.
                            It is used to measure the robustness of the model
        """
        N = len(self.y_real)
        Logistic_Loss = -(1 / N) * sum(
            self.y_real * np.log(self.y_pre_prob) + (1 - self.y_real) * np.log(1 - self.y_pre_prob))
        return Logistic_Loss

    def auc_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            AUC ——  an extensively used evaluation measure obtained from
                    the Receiver Operating Characteristic (ROC) curve.
                    representing the area under the ROC curve. ROC curve:
                    The x- axis represents the false-positive rate (computed as 1-specificity)
                    the y-axis represents true-positive rate sensitivity)
        """
        FPR, TPR, thresholds = roc_curve(self.y_real, self.y_pre_prob)
        AUC = auc(FPR, TPR)
        return AUC, FPR, TPR

    def mcc_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            MCC ——  马休斯相关系数MCC (Matthews correlation coefficient):
                A correlation of:
                C = 1 indicates perfect agreement,
                C = 0 is expected for a prediction no better than random, and
                C = -1 indicates total disagreement between prediction and observation
                系数为 1  的时候，分类器是完美;
                系数为 0  的时候分类器和随机分类器没差别;
                系数为 -1 的时候分类器是最差的，所有预测结果和实际相反.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        # from sklearn.metrics import matthews_corrcoef
        # matthews_corrcoef(y_real, y_pred_label)
        return MCC

    def ks_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            KS ——  柯尔莫诺夫-斯米尔诺夫值
                (Kolmogorov-Smirnov)值越大，表示模型能够将正、负客户区分开的程度越大。
                KS值的取值范围是[0，1]. 通常来讲，KS>0.2即表示模型有较好的预测准确性.
        """
        temp, FPR, TPR = self.auc_ykp()
        KS = max(TPR - FPR)
        return KS

    def bm_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            BM ——  Bookmaker Informedness (BM) Informedness.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        BM = TP / (TP + FN) + TN / (TN + FP) - 1
        return BM

    def mk_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            MK ——  Markedness (MK).
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        MK = TP / (TP + FP) + TN / (TN + FN) - 1
        return MK

    def Brier_score_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_label   —— 预测得到的标签值
        Output:
            Brier_score ——  Brier score (BS).是误差的概念, 布莱尔分数(BS)越小越好.
        """
        Brier_score = sum(np.power((self.y_real-self.y_pre_label), 2)/len(self.y_real))
        return Brier_score

    def Kappa_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Kappa ——  Kappa值越大,预测效果越好.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        P_o = (TP + TN) / (TP + FP + FN + TN)
        P_e = ((TP + FN) * (TP + FP) + (FN + TN) * (FP + TN)) / ((TP + FP + FN + TN) * (TP + FP + FN + TN))
        Kappa = (P_o-P_e)/(1-P_e)
        return Kappa

    def Inverse_precision_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Inverse_precision ——  Inverse_precision(预测为非违约,且真实也是非违约,的正确率)
            Measures proportion of correctly classified negative cases out of total negative predictions.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Inverse_precision = TN/(TN+FN)
        return Inverse_precision

    def Jaccard_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Jaccard  ——  Jaccard coefficient (Measures similarity between actual and predicted values)
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Jaccard = TP / (TP+FP+FN)
        return Jaccard

    def Youden_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            Youden  ——  Youden’s Index(Measures discriminating power of the test i.e. ability of classifier to avoid misclassification)
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        Youden = TP/(TP+FN)+TN/(TN+FP)-1
        return Youden

    def gini_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            gini ——  defined as twice the area between the ROC curve and the chance diagona.
        """
        FPR, TPR, thresholds = roc_curve(self.y_real, self.y_pre_prob)
        AUC = auc(FPR, TPR)
        gini = 2*AUC - 1
        return gini

    def accuracy_dict_ykp(self):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_prob   —— 预测标签概率值
            y_pre_label —— 预测得到的标签值
        Output:
            Accuracy_Output:
                TP,TN,FP,FN   —— 混淆矩阵结果
                Accuracy      —— 整体精度
                Precision     —— 准确率
                Recall        —— 召回率
                F_measure     —— F指数
                Sensitivity   —— 敏感度
                Specificity   —— 特异度
                G_Means       —— G-means几何平均值
                Logistic_Loss —— 逻辑损失值
                AUC           —— ROC曲线下的面积
                MCC           —— 马休斯相关系数
                KS            —— KS值
                BM            —— Bookmaker Informedness
                MK            —— Markedness
                Type_II_error —— = 1-Recall,是第二类错误(将违约样本错判为非违约的比例)
                Type_I_error  —— = 1-Specificity,是第一类错误(将非违约样本错判为违约的比例)
                Brier_score   —— 布莱尔分数Brier score (BS).
                Kappa         ——  Kappa值越大,预测效果越好.
                Inverse_precision ——  Inverse_precision(预测为非违约,且真实也是非违约,的正确率)
                Jaccard  ——  Jaccard coefficient (Measures similarity between actual and predicted values)
                Youden  ——  Youden’s Index(Measures discriminating power of the test i.e. ability of classifier to avoid misclassification)
                gini ——  基尼系数.defined as twice the area between the ROC curve and the chance diagona.
        """
        Accuracy_dict = dict()
        Accuracy_dict['TN'] = self.confusion_matrix_ykp()[0]
        Accuracy_dict['FP'] = self.confusion_matrix_ykp()[1]
        Accuracy_dict['FN'] = self.confusion_matrix_ykp()[2]
        Accuracy_dict['TP'] = self.confusion_matrix_ykp()[3]
        Accuracy_dict['Precision'] = self.f_measure_ykp()[0]
        Accuracy_dict['Recall'] = self.f_measure_ykp()[1]
        Accuracy_dict['Accuracy'] = self.accuracy_ykp()
        Accuracy_dict['F_measure'] = self.f_measure_ykp()[2]
        Accuracy_dict['G_Means'] = self.g_means_ykp()[2]
        Accuracy_dict['Logistic_Loss'] = self.logistic_loss_ykp()
        Accuracy_dict['AUC'] = self.auc_ykp()[0]
        Accuracy_dict['MCC'] = self.mcc_ykp()
        Accuracy_dict['KS'] = self.ks_ykp()
        Accuracy_dict['BM'] = self.bm_ykp()
        Accuracy_dict['MK'] = self.mk_ykp()
        Accuracy_dict['Type_II_error'] = self.f_measure_ykp()[3]
        Accuracy_dict['Type_I_error'] = self.g_means_ykp()[3]
        Accuracy_dict['Brier_score'] = self.Brier_score_ykp()
        Accuracy_dict['Kappa'] = self.Kappa_ykp()
        Accuracy_dict['Inverse_precision'] = self.Inverse_precision_ykp()
        Accuracy_dict['Jaccard'] = self.Jaccard_ykp()
        Accuracy_dict['Youden'] = self.Youden_ykp()
        Accuracy_dict['gini'] = self.gini_ykp()
        return Accuracy_dict


# 模型训练及预测
class BasePredictYkp:
    def __init__(self, X_train, Y_train, X_test, Y_test, model_list):
        """
        Input variables:
            X_train     —— 训练样本
            Y_train     —— 训练样本的0-1标签(1-为违约)
            X_test      —— 测试样本
            Y_test      —— 测试样本的0-1标签(1-为违约)
            model_list  —— 分类模型的类型,包括['LG', 'SVM', 'LDA', 'DT', 'KNN', 'ANN']
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model_list = model_list

    def multi_class_model_ykp(self, Model):
        # single model predict
        """
        Input variables:
            X_train —— 训练样本
            Y_train —— 训练样本的0-1标签(1-为违约)series
            X_test  —— 测试样本
            Y_test  —— 测试样本的0-1标签(1-为违约)series
            Model   —— 分类模型的类型,包括
                       model_list = ['SVM','LG','NB','DT','KNN','RF','LDA','GBDT','ANN','Ada']
        To use this function, coding:
            case, y_pre_prob, y_pre_label, accuracy = multi_class_model_ykp(X_train, Y_train, X_test, Y_test, 'SVM')
        """
        case = []
        y_pre_prob = []
        y_pre_label = []
        accuracy = dict()
        if Model == 'SVM':
            ''' 01.Linear_SVM 模型检验精度 '''
            # clf_SVM = svm.SVC(C=0.8, kernel='rbf', gamma=20, probability=True, decision_function_shape='ovr')
            clf_SVM = svm.SVC(kernel='linear', probability=True)
            case_SVM = clf_SVM.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_SVM = case_SVM.predict_proba(self.X_test)[:, 1]
            y_pre_label_SVM = case_SVM.predict(self.X_test)
            accuracy_SVM = AccuracyMeasureYkp(self.Y_test, y_pre_prob_SVM, y_pre_label_SVM)
            # 保存
            case = case_SVM
            y_pre_prob = y_pre_prob_SVM
            y_pre_label = y_pre_label_SVM
            accuracy = accuracy_SVM.accuracy_dict_ykp()
        elif Model == 'LG':
            '''  02.Logistics_regression检验精度'''
            """
            clf_LG = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=0.008, fit_intercept=True,
                                        intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
                                        max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            """
            clf_LG = LogisticRegression()
            case_LG = clf_LG.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_LG = case_LG.predict_proba(self.X_test)[:, 1]
            y_pre_label_LG = case_LG.predict(self.X_test)
            accuracy_LG = AccuracyMeasureYkp(self.Y_test, y_pre_prob_LG, y_pre_label_LG)
            # 保存
            case = case_LG
            y_pre_prob = y_pre_prob_LG
            y_pre_label = y_pre_label_LG
            accuracy = accuracy_LG.accuracy_dict_ykp()
        elif Model == 'NB':
            ''' 03.NB 高斯朴素贝叶斯检验精度 '''
            clf_NB = GaussianNB()
            case_NB = clf_NB.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_NB = case_NB.predict_proba(self.X_test)[:, 1]
            y_pre_label_NB = case_NB.predict(self.X_test)
            accuracy_NB = AccuracyMeasureYkp(self.Y_test, y_pre_prob_NB, y_pre_label_NB)
            # 保存
            case = case_NB
            y_pre_prob = y_pre_prob_NB
            y_pre_label = y_pre_label_NB
            accuracy = accuracy_NB.accuracy_dict_ykp()
        elif Model == 'DT':
            ''' 04.DT决策树检验精度 '''
            clf_DT = DecisionTreeClassifier(max_depth=4)
            case_DT = clf_DT.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_DT = case_DT.predict_proba(self.X_test)[:, 1]
            y_pre_label_DT = case_DT.predict(self.X_test)
            accuracy_DT = AccuracyMeasureYkp(self.Y_test, y_pre_prob_DT, y_pre_label_DT)
            # 保存
            case = case_DT
            y_pre_prob = y_pre_prob_DT
            y_pre_label = y_pre_label_DT
            accuracy = accuracy_DT.accuracy_dict_ykp()
        elif Model == 'KNN':
            ''' 05.KNN算法检验精度 '''
            clf_KNN = KNeighborsClassifier()
            case_KNN = clf_KNN.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_KNN = case_KNN.predict_proba(self.X_test)[:, 1]
            y_pre_label_KNN = case_KNN.predict(self.X_test)
            accuracy_KNN = AccuracyMeasureYkp(self.Y_test, y_pre_prob_KNN, y_pre_label_KNN)
            # 保存
            case = case_KNN
            y_pre_prob = y_pre_prob_KNN
            y_pre_label = y_pre_label_KNN
            accuracy = accuracy_KNN.accuracy_dict_ykp()
        elif Model == 'RF':
            ''' 06.RF随机森林分类器检验精度 '''
            clf_RF = RandomForestClassifier()
            case_RF = clf_RF.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_RF = case_RF.predict_proba(self.X_test)[:, 1]
            y_pre_label_RF = case_RF.predict(self.X_test)
            accuracy_RF = AccuracyMeasureYkp(self.Y_test, y_pre_prob_RF, y_pre_label_RF)
            # 保存
            case = case_RF
            y_pre_prob = y_pre_prob_RF
            y_pre_label = y_pre_label_RF
            accuracy = accuracy_RF.accuracy_dict_ykp()
        elif Model == 'LDA':
            ''' 07.LDA线性判别法检验精度 '''
            clf_LDA = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                                 store_covariance=False, tol=0.01)
            case_LDA = clf_LDA.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_LDA = case_LDA.predict_proba(self.X_test)[:, 1]
            y_pre_label_LDA = case_LDA.predict(self.X_test)
            accuracy_LDA = AccuracyMeasureYkp(self.Y_test, y_pre_prob_LDA, y_pre_label_LDA)
            # 保存
            case = case_LDA
            y_pre_prob = y_pre_prob_LDA
            y_pre_label = y_pre_label_LDA
            accuracy = accuracy_LDA.accuracy_dict_ykp()
        elif Model == 'GBDT':
            ''' 08.GBDT梯度提升树检验精度 '''
            clf_GBDT = GradientBoostingClassifier()
            case_GBDT = clf_GBDT.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_GBDT = case_GBDT.predict_proba(self.X_test)[:, 1]
            y_pre_label_GBDT = case_GBDT.predict(self.X_test)
            accuracy_GBDT = AccuracyMeasureYkp(self.Y_test, y_pre_prob_GBDT, y_pre_label_GBDT)
            # 保存
            case = case_GBDT
            y_pre_prob = y_pre_prob_GBDT
            y_pre_label = y_pre_label_GBDT
            accuracy = accuracy_GBDT.accuracy_dict_ykp()
        elif Model == 'ANN':
            ''' 09.ANN神经网络模型检验精度 '''
            clf_ANN = MLPClassifier(hidden_layer_sizes=[4], activation='logistic', solver='lbfgs', random_state=0)
            case_ANN = clf_ANN.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_ANN = case_ANN.predict_proba(self.X_test)[:, 1]
            y_pre_label_ANN = case_ANN.predict(self.X_test)
            accuracy_ANN = AccuracyMeasureYkp(self.Y_test, y_pre_prob_ANN, y_pre_label_ANN)
            # 保存
            case = case_ANN
            y_pre_prob = y_pre_prob_ANN
            y_pre_label = y_pre_label_ANN
            accuracy = accuracy_ANN.accuracy_dict_ykp()
        elif Model == 'Ada':
            ''' 10.Ada-boost模型检验精度 '''
            clf_Ada = AdaBoostClassifier()
            case_Ada = clf_Ada.fit(self.X_train, self.Y_train.ravel())
            # 预测
            y_pre_prob_Ada = case_Ada.predict_proba(self.X_test)[:, 1]
            y_pre_label_Ada = case_Ada.predict(self.X_test)
            accuracy_Ada = AccuracyMeasureYkp(self.Y_test, y_pre_prob_Ada, y_pre_label_Ada)
            # 保存
            case = case_Ada
            y_pre_prob = y_pre_prob_Ada
            y_pre_label = y_pre_label_Ada
            accuracy = accuracy_Ada.accuracy_dict_ykp()
        else:
            print("\n 【Warning】: 'Model' input wrongly, no such model!!!\n")
        """
        Output variables:
            case         —— 训练集上得到的线性 Model 模型
            y_pre_prob   —— 测试集上 Model 预测得到的 预测概率值
            y_pre_label  —— 测试集上 Model 预测得到的 预测标签值
            accuracy     —— 测试集上 Model 预测得到的精度判别的dict类别变量
        """
        return case, y_pre_prob, y_pre_label, accuracy

    def single_predict_out_ykp(self):
        # 先计算在训练集上训练模型，在测试集上得到的结果, 并保存至 dict() 中
        case_train_dict = dict()
        y_test_pre_prob_df = pd.DataFrame()
        y_test_pre_label_df = pd.DataFrame()
        accuracy_test_dict = dict()
        accuracy_test_df = pd.DataFrame()
        for i in self.model_list:
            # print(i)
            locals()['case_train_%s' % i], locals()['y_test_pre_prob_%s' % i], locals()[
                'y_test_pre_label_%s' % i], locals()[
                'accuracy_test_%s' % i] = self.multi_class_model_ykp(i)
            case_train_dict[i] = locals()['case_train_%s' % i]
            y_test_pre_prob_df[i+'_概率值'] = locals()['y_test_pre_prob_%s' % i]
            y_test_pre_label_df[i+'_标签值'] = locals()['y_test_pre_label_%s' % i]
            accuracy_test_dict[i] = locals()['accuracy_test_%s' % i]
            accuracy_test_df[i+'_精度结果'] = pd.DataFrame(list(accuracy_test_dict[i].items())).iloc[:, 1]
        # temp_df = pd.DataFrame(list(accuracy_test_dict[0].items()))
        accuracy_test_df.index = list(pd.DataFrame(list(accuracy_test_dict[self.model_list[-1]].items())).iloc[:, 0])
        return y_test_pre_prob_df, y_test_pre_label_df, accuracy_test_df, case_train_dict


def save_with_pickle_ykp(save_path, save_name, save_obj):
    # save variables in pickle-file after running codes
    """
    Input variables:
        save_path  ——  pickle文件的保存路径
        save_name  ——  pickle文件的名称
        save_obj   ——  pickle文件中所要保存的变量名,eg: [ {'data_features':data_features,'data_label':data_label} ]
    To use this function, coding:
        save_with_pickle_ykp( save_path, save_name, save_obj )
    """
    with open(save_path + save_name + '.p', "wb") as file_out:
        pickle.dump(save_obj, file_out)
    """
    Output variables:
        保存的pickle文件详见: 对应于save_path路径下的pickle文件
    """


def load_pickle_ykp(save_path, save_name):
    # load results in pickle-file
    """
    Input variables:
        save_path  ——  pickle文件的保存路径
        save_name  ——  pickle文件的名称
    To use this function, coding:
        load_pickle_ykp(save_path, save_name)
    """
    with open(save_path + save_name + '.p', "rb") as file_in:
        all_results = pickle.load(file_in)
    """
    Output variables:
        all_results —— 将save_path路径下的pickle文件内容读取出来
    """
    return all_results


def main():
    # 以下6行是需自行设置的参数
    excel_file_name_train = "Data_in.xlsx"  # 可替换成自己想读取的-训练-数据文件名+后缀
    excel_sheet_name_train = "Sheet_train"  # 可替换成自己想读取的-训练-数据文件中的sheet名
    excel_file_name_test = "Data_in.xlsx"  # 可替换成自己想读取的-测试-数据文件名+后缀
    excel_sheet_name_test = "Sheet_test"  # 可替换成自己想读取的-测试-数据文件中的sheet名
    model_list = ["LG", "RF"]  # 可替换成自己想使用的模型名称, 名称缩写需从BasePredictYkp()已有的选择
    output_excel_file_name = "Data_out.xlsx"  # 可自定义为想输出的excel文件名+后缀
    # 以下不用人工修改
    code_path = os.getcwd() + '\\'  # 获取当前代码的路径
    # 训练数据读取
    data_in_train = pd.read_excel(code_path+excel_file_name_train, sheet_name=excel_sheet_name_train)  # 读取所有训练数据
    data_in_x_train = data_in_train.iloc[:, 0: -1]  # 读取除最后一列外的所有列-训练
    data_in_y_train = data_in_train.iloc[:, -1]  # 读取最后一列标签-训练
    # 测试数据读取
    data_in_test = pd.read_excel(code_path + excel_file_name_test, sheet_name=excel_sheet_name_test)  # 读取所有测试数据
    data_in_x_test = data_in_test.iloc[:, 0: -1]  # 读取除最后一列外的所有列-测试
    data_in_y_test = data_in_test.iloc[:, -1]  # 读取最后一列标签-测试
    # 训练集训练模型,在训练集上的预测结果
    base_train_predict_define = BasePredictYkp(data_in_x_train, data_in_y_train, data_in_x_train, data_in_y_train,
                                               model_list)
    base_train_predict_out = base_train_predict_define.single_predict_out_ykp()  # 第1-4项分别是:预测概率,预测标签,预测精度,训练的模型
    # 训练集训练的模型,在测试集上的预测结果
    base_test_predict_define = BasePredictYkp(data_in_x_train, data_in_y_train, data_in_x_test, data_in_y_test,
                                              model_list)
    base_test_predict_out = base_test_predict_define.single_predict_out_ykp()  # 第1-4项分别是:预测概率,预测标签,预测精度,训练的模型
    # 结果excel输出
    now_time_ymdh = datetime.datetime.now().strftime('%Y.%m.%d.%H')  # 记录当前的年、月、日、小时
    now_time_m = datetime.datetime.now().strftime('%M')  # 记录当前的分钟
    writer = pd.ExcelWriter(code_path + now_time_ymdh + "：" + now_time_m + "." + output_excel_file_name)
    base_train_predict_out[0].to_excel(writer, sheet_name="01.预测的概率值(训练集)", index=True)
    base_train_predict_out[1].to_excel(writer, sheet_name="02.预测的标签值(训练集)", index=True)
    base_train_predict_out[2].to_excel(writer, sheet_name="03.预测的精度值(训练集)", index=True)
    base_test_predict_out[0].to_excel(writer, sheet_name="04.预测的概率值(测试集)", index=True)
    base_test_predict_out[1].to_excel(writer, sheet_name="05.预测的标签值(测试集)", index=True)
    base_test_predict_out[2].to_excel(writer, sheet_name="06.预测的精度值(测试集)", index=True)
    writer.save()
    return base_train_predict_out, base_test_predict_out, \
           excel_file_name_train, excel_sheet_name_train, \
           excel_file_name_test, excel_sheet_name_test, \
           model_list, output_excel_file_name


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    time_start_all = time.time()
    print('\n 程序开始运行...')
    Data_output_final = main()
    # 将变量设置及结果同时保存至pickle文件中,以便之后查询模型参数及调用训练好的模型
    nowTime1_final = datetime.datetime.now().strftime('%Y.%m.%d.%H')  # 记录当前的年、月、日、小时
    nowTime2_final = datetime.datetime.now().strftime('%M')  # 记录当前的分钟
    save_name_final = nowTime1_final + '：' + nowTime2_final + '.保存的模型所有变量结果'
    save_obj_final = {'Data_output': Data_output_final}
    save_with_pickle_ykp(os.getcwd() + '\\', save_name_final, save_obj_final)
    time_end_all = time.time()
    print('\n 所有程序运行成功,并保存完毕,  cost time: ', (time_end_all - time_start_all) / 60, ' 分钟.\n')
