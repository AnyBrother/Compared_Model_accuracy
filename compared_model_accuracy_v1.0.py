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
            1.2 新增1个二分类的精度标准.(Added: H_measure)
            1.3 新增32个三分类的精度标准.(Added: F1, F1_micro, F1_macro, F1_weighted, etc.)
"""
import os
import pandas as pd
import numpy as np
from sympy import *


from sklearn.metrics import roc_curve, auc, confusion_matrix  # 导入 获取两分类 精度判别的计算包
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, \
    f1_score, precision_score, recall_score, roc_auc_score, jaccard_score  # 导入 获取多分类 精度判别的计算包
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
            weight       —— 两类0:1的非平衡比例
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
            self.y_real * np.log(self.y_pre_prob+1e-20) + (1 - self.y_real) * np.log(1 - self.y_pre_prob+1e-20))
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
            Youden  ——  Youden’s Index(Measures discriminating power of the test
                        i.e. ability of classifier to avoid misclassification)
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

    @staticmethod
    def beta_distribution(x, a, b):
        # beta distribution value
        t = symbols('t')
        beta_a_b = integrate(t ** (a - 1) * (1 - t) ** (b - 1), (t, 0, 1))
        beta_x = 1 / beta_a_b * x ** (a - 1) * (1 - x) ** (b - 1)
        return beta_x

    def h_measure_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            h_measure ——  defined as an effective metric that provides
                            coherent misclassification cost to evaluate the competence of models.
        """
        # beta分布的参数设置,默认是beta(2,2)
        a = 2
        b = 2
        # 开始计算h_measure
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        pai_1 = (TP+FN)/(TP+FN+TN+FP)   # 真实违约客户占所有客户的比重
        pai_0 = (TN+FP)/(TP+FN+TN+FP)   # 真实非违约客户占所有客户的比重
        x = symbols('x')
        # beta_integrate_value = integrate(self.beta_distribution(x, a, b), (x, range_min, range_max))
        L_max = pai_0*integrate(x*self.beta_distribution(x, a, b), (x, 0, pai_1))+pai_1*integrate((1-x)*self.beta_distribution(x, a, b), (x, pai_1, 1))
        L = pai_0 * integrate(x * FP/(TP+FN+TN+FP), (x, 0, pai_1)) + pai_1 * integrate((1 - x) * FN/(TP+FN+TN+FP), (x, pai_1, 1))
        # L = integrate((x*FP/(TP+FN+TN+FP)+(1-x)*FN/(TP+FN+TN+FP))*self.beta_distribution(x, a, b), (x, 0, 1))
        h_measure = 1-L/L_max
        return h_measure

    def Weighted_cross_entropy(self):
        """
        :param y_pre_probs:  net's output, which has reshaped [batch size,num_class]
        :param label:   Ground Truth which is ont hot encoing and has typr format of [batch size, num_class]
        :param weight:  a vector that describes every catagory's coefficent whose shape is (num_class,)
        :return: a scalar
        """
        weight = [1, len(self.y_real)/self.y_real.sum()-1]
        label = np.c_[1-np.array(self.y_real), np.array(self.y_real)]
        y_pre_probs = np.c_[1-np.array(self.y_pre_prob+1e-20), np.array(self.y_pre_prob+1e-20)]
        loss = np.dot(np.log2(y_pre_probs) * label, np.expand_dims(weight, axis=1)) + \
               np.log2(y_pre_probs) * (1 - label)
        return -loss.sum()

    def focal_loss(self, a=0.5, r=0.9):
        """
        :param y_pre_probs: [batch size,num_classes] score value
        :param label: [batch size,num_classes] gt value
        :param a: generally be 0.5. 样本平衡因子，在0-1之间
        :param r: generally be 0.9. 作用是相对放大难分类样本的梯度，相对降低易分类样本的梯度，为0时则是标准的交叉熵
        :return: scalar loss value of a batch
        """
        label = np.c_[1 - np.array(self.y_real), np.array(self.y_real)]
        y_pre_probs = np.c_[1 - np.array(self.y_pre_prob + 1e-20), np.array(self.y_pre_prob + 1e-20)]
        p_1 = - a * np.power(1 - y_pre_probs, r) * np.log2(y_pre_probs) * label
        p_0 = - (1 - a) * np.power(y_pre_probs, r) * np.log2(1 - y_pre_probs) * (1 - label)
        return (p_1 + p_0).sum()

    def dice_loss(self):
        """
        dice_loss 来自文章V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation，
                    旨在应对语义分割中正负样本强烈不平衡的场景.
        dice_coefficient = 2*TP/(FP+FN+2*TP)
        dice_coefficient 是可以体现出预测区域和真实区域的重叠程度，它的取值范围是[0, 1]，
                        当dice coefficient为1时，说明预测区域和真实区域完全重叠，是理想状态；
                        当dice coefficient为0时，说明预测结果一点作用没有。
        dice_loss = 1 - dice_coefficient = 1 - 2*TP/(FP+FN+2*TP). 为避免分子分母0的出现, 一般分子分母都加一个极小数.
                    由于是离散值, 因此需要用到预测的概率值, 使得损失函数连续.
                    此处, 只用于目标效果验证, 因此, 取离散形式.
        """
        TN, FP, FN, TP = self.confusion_matrix_ykp()
        dice_loss = 1 - 2 * TP / (FP + FN + 2 * TP)
        return dice_loss

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
                H_measure——H值.an effective metric that provides coherent misclassification cost to evaluate the competence of models.
                Weighted_cross_entropy_loss——加权交叉熵损失. 常用于深度学习中的损失函数.
                Focal_loss——何凯明大神的RetinaNet中提出了Focal Loss来解决类别不平衡的问题. 常用于深度学习中的非平衡损失函数. 更关注与难分类的样本
                Dice_loss——用于处理CNN中的非平衡样本. https://arxiv.org/abs/1606.04797
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
        Accuracy_dict['H_measure'] = self.h_measure_ykp()
        Accuracy_dict['Weighted_cross_entropy_loss'] = self.Weighted_cross_entropy()
        Accuracy_dict['Focal_loss'] = self.focal_loss(a=0.5, r=0.9)
        Accuracy_dict['Dice_loss'] = self.dice_loss()
        return Accuracy_dict


class AccuracyMeasureMultiYkp:
    # 计算多分类的精度结果
    """
        ['Accuracy', 'F1_macro', 'F1_micro', 'F1_weighted',
        'Sensitivity_macro', 'Sensitivity_micro', 'Sensitivity_weighted',
        'Specificity_macro', 'Specificity_micro', 'Specificity_weighted',
        'G_Means_macro', 'G_Means_micro', 'G_Means_weighted',
        'Type_I_error_macro', 'Type_I_error_micro', 'Type_I_error_weighted',
        'Type_II_error_macro', 'Type_II_error_micro', 'Type_II_error_weighted',
        'Precision_macro', 'Precision_micro', 'Precision_weighted',
        'Recall_macro', 'Recall_micro', 'Recall_weighted',
        'AUC_ovr_macro', 'AUC_ovr_weighted', 'AUC_ovo_macro', 'AUC_ovo_weighted',
        'Jaccard_macro', 'Jaccard_micro', 'Jaccard_weighted']
    """
    def __init__(self, y_real, y_pre_label, y_pre_prob, y_pre_prob_threshold=0.5):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_prob   —— 预测标签概率值
            y_pre_label  —— 预测得到的标签值
        """
        self.y_real = y_real
        self.y_pre_label = y_pre_label
        self.y_pre_prob = y_pre_prob
        self.y_pre_prob_threshold = y_pre_prob_threshold

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
        cf_matrix = confusion_matrix(self.y_real, self.y_pre_label)
        mlcf_matrix = multilabel_confusion_matrix(self.y_real, self.y_pre_label)
        return cf_matrix, mlcf_matrix

    def accuracy_ykp(self):
        """
        Input:
            TP,TN,FP,FN  —— 混淆矩阵结果
        Output:
            Accuracy     ——  One of the prevailing evaluation measure
                        and defined as the correct prediction sample size
                        divided by the total testing sample size.
        """
        Accuracy = accuracy_score(self.y_real, self.y_pre_label)
        return Accuracy

    def f_measure_ykp(self):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_label  —— 预测得到的标签值
        Output:
            F1_Macro     —— 称为F1分数或F分数,是权衡Precision和Recall是使用精度和召回率的方法组合到一个度量上
        Reference:
            Zhou. 2017. http://dx.doi.org/10.1016/j.knosys.2017.05.003
        """
        F1_macro = f1_score(self.y_real, self.y_pre_label, average='macro')  # 取所有类的平均值
        F1_micro = f1_score(self.y_real, self.y_pre_label, average='micro')  # 再取所有类的micro值
        F1_weighted = f1_score(self.y_real, self.y_pre_label, average='weighted')  # 取所有类的样本比例的weighted值
        return F1_macro, F1_micro, F1_weighted

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
        mlcf_matrix = self.confusion_matrix_ykp()[1]
        # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
        tn_sum = mlcf_matrix[:, 0, 0]  # True Negative
        fp_sum = mlcf_matrix[:, 0, 1]  # False Positive
        tp_sum = mlcf_matrix[:, 1, 1]  # True Positive
        fn_sum = mlcf_matrix[:, 1, 0]  # False Negative
        # 计算Sensitivity
        Condition_negative_se = tp_sum + fn_sum
        Condition_negative_se = np.array([1e-6 if x == 0 else x for x in Condition_negative_se])  # 这里加1e-6，防止 0/0的情况
        Sensitivity = tp_sum / Condition_negative_se  # Sensitivity = TP / (TP + FN)
        Sensitivity_macro = np.average(Sensitivity, weights=None)
        Sensitivity_micro = np.sum(tp_sum) / np.sum(tp_sum + fn_sum)
        Sensitivity_weighted = np.average(Sensitivity, weights=Condition_negative_se)
        # 计算Specificity
        Condition_negative_sp = tn_sum + fp_sum
        Condition_negative_sp = np.array([1e-6 if x == 0 else x for x in Condition_negative_sp])  # 这里加1e-6，防止 0/0的情况
        Specificity = tn_sum / Condition_negative_sp  # Specificity = TN / (TN + FP)
        Specificity_macro = np.average(Specificity, weights=None)
        Specificity_micro = np.sum(tn_sum) / np.sum(tn_sum + fp_sum)
        Specificity_weighted = np.average(Specificity, weights=Condition_negative_sp)
        # 计算G-Mean
        G_Means = np.sqrt(Sensitivity * Specificity)  # G_Means = np.sqrt(Sensitivity * Specificity)
        G_Means_macro = np.average(G_Means, weights=None)
        G_Means_micro = np.sum(G_Means) / np.sum(G_Means)
        G_Means_weighted = np.average(G_Means, weights=Condition_negative_sp+Condition_negative_se)
        # 计算Type_I_error
        Type_I_error = 1 - Specificity  # 第一类错误(将好样本错判为差样本的比例)
        Type_I_error_macro = np.average(Type_I_error, weights=None)
        Type_I_error_micro = np.sum(Type_I_error) / np.sum(Type_I_error)
        Type_I_error_weighted = np.average(Type_I_error, weights=Condition_negative_sp)
        return Sensitivity_macro, Sensitivity_micro, Sensitivity_weighted, \
               Specificity_macro, Specificity_micro, Specificity_weighted, \
               G_Means_macro, G_Means_micro, G_Means_weighted, \
               Type_I_error_macro, Type_I_error_micro, Type_I_error_weighted

    def precision_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            precision     ——  预测的1中判对的比例
        """
        Precision_macro = precision_score(self.y_real, self.y_pre_label, average='macro')  # precision = TP / (TP + FP)
        Precision_micro = precision_score(self.y_real, self.y_pre_label, average='micro')  # precision = TP / (TP + FP)
        Precision_weighted = precision_score(self.y_real, self.y_pre_label, average='weighted')  # precision = TP / (TP + FP)
        return Precision_macro, Precision_micro, Precision_weighted

    def recall_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值
        Output:
            recall_score     ——  真实的1中判对的比例
            Type_II_error    —— 第二类错误(将差样本错判为好样本的比例)
        """
        Recall_macro = recall_score(self.y_real, self.y_pre_label, average='macro')  # Recall = TP / (TP + FN)
        Recall_micro = recall_score(self.y_real, self.y_pre_label, average='micro')  # Recall = TP / (TP + FN)
        Recall_weighted = recall_score(self.y_real, self.y_pre_label, average='weighted')  # Recall = TP / (TP + FN)
        # Type_II_error = 1 - Recall  # 第二类错误(将差样本错判为好样本的比例)
        Type_II_error_macro = 1 - Recall_macro
        Type_II_error_micro = 1 - Recall_micro
        Type_II_error_weighted = 1 - Recall_weighted
        return Recall_macro, Recall_micro, Recall_weighted, \
               Type_II_error_macro, Type_II_error_micro, Type_II_error_weighted

    def auc_ykp(self):
        """
        Input:
            y_real        —— 真实标签值
            y_pre_prob    —— 预测得到的概率值. 多分类必须是每个标签对应的概率
        Output:
            AUC ——  an extensively used evaluation measure obtained from
                    the Receiver Operating Characteristic (ROC) curve.
                    representing the area under the ROC curve. ROC curve:
                    The x-axis represents the false-positive rate (computed as 1-specificity)
                    the y-axis represents true-positive rate sensitivity)
        """
        # ovr对类别不平衡比较敏感
        AUC_ovr_macro = roc_auc_score(np.array(self.y_real), self.y_pre_prob, average='macro', multi_class='ovr')
        AUC_ovr_weighted = roc_auc_score(np.array(self.y_real), self.y_pre_prob, average='weighted', multi_class='ovr')
        # ovo对类别不平衡不敏感
        AUC_ovo_macro = roc_auc_score(np.array(self.y_real), self.y_pre_prob, average='macro', multi_class='ovo')
        AUC_ovo_weighted = roc_auc_score(np.array(self.y_real), self.y_pre_prob, average='weighted', multi_class='ovo')
        return AUC_ovr_macro, AUC_ovr_weighted, AUC_ovo_macro, AUC_ovo_weighted

    def jaccard_ykp(self):
        """
        Input:
            TP,TN,FP,FN   —— 混淆矩阵结果
        Output:
            MCC ——  马Jaccard 相似系数得分.
                定义为交集的大小除以两个标签集的并集大小，用于将样本的预测标签集与对应的标签集进行比较y_true 。
        """
        Jaccard_macro = jaccard_score(self.y_real, self.y_pre_label, average='macro')
        Jaccard_micro = jaccard_score(self.y_real, self.y_pre_label, average='micro')
        Jaccard_weighted = jaccard_score(self.y_real, self.y_pre_label, average='weighted')
        return Jaccard_macro, Jaccard_micro, Jaccard_weighted

    def accuracy_dict_ykp(self):
        """
        Input:
            y_real       —— 真实标签值
            y_pre_prob   —— 预测标签概率值
            y_pre_label —— 预测得到的标签值
        Output:
            Accuracy_Output:
                cf_matrix, mlcf_matrix   —— 混淆矩阵结果
                Accuracy      —— 准确率 (Accuracy)
                F_measure     —— F指数 (F1_macro, F1_micro, F1_weighted)
                Sensitivity   —— 敏感度 (Sensitivity_macro, Sensitivity_micro, Sensitivity_weighted)
                Specificity   —— 特异度 (Specificity_macro, Specificity_micro, Specificity_weighted)
                G_Mean        —— G-means几何平均值 (G_Means_macro, G_Means_micro, G_Means_weighted)
                Type_I_error  —— = 1-Specificity,是第一类错误(将非违约样本错判为违约的比例)
                                    (Type_I_error_macro, Type_I_error_micro, Type_I_error_weighted)
                Type_II_error —— = 1-Recall,是第二类错误(将违约样本错判为非违约的比例)
                                    (Type_II_error_macro, Type_II_error_micro, Type_II_error_weighted)
                Precision     —— 准确率 (Precision_macro, Precision_micro, Precision_weighted)
                Recall        —— 召回率 (Recall_macro, Recall_micro, Recall_weighted)
                AUC           —— ROC曲线下的面积 (AUC_ovr_macro, AUC_ovr_weighted, AUC_ovo_macro, AUC_ovo_weighted)
                Jaccard       —— Jaccard相似系数 (Jaccard_macro, Jaccard_micro, Jaccard_weighted)
        """
        Accuracy_dict = dict()
        Accuracy_dict['Accuracy'] = self.accuracy_ykp()
        Accuracy_dict['F1_macro'] = self.f_measure_ykp()[0]
        Accuracy_dict['F1_micro'] = self.f_measure_ykp()[1]
        Accuracy_dict['F1_weighted'] = self.f_measure_ykp()[2]
        Accuracy_dict['Sensitivity_macro'] = self.g_means_ykp()[0]
        Accuracy_dict['Sensitivity_micro'] = self.g_means_ykp()[1]
        Accuracy_dict['Sensitivity_weighted'] = self.g_means_ykp()[2]
        Accuracy_dict['Specificity_macro'] = self.g_means_ykp()[3]
        Accuracy_dict['Specificity_micro'] = self.g_means_ykp()[4]
        Accuracy_dict['Specificity_weighted'] = self.g_means_ykp()[5]
        Accuracy_dict['G_Means_macro'] = self.g_means_ykp()[6]
        Accuracy_dict['G_Means_micro'] = self.g_means_ykp()[7]
        Accuracy_dict['G_Means_weighted'] = self.g_means_ykp()[8]
        Accuracy_dict['Type_I_error_macro'] = self.g_means_ykp()[9]
        Accuracy_dict['Type_I_error_micro'] = self.g_means_ykp()[10]
        Accuracy_dict['Type_I_error_weighted'] = self.g_means_ykp()[11]
        Accuracy_dict['Type_II_error_macro'] = self.recall_ykp()[3]
        Accuracy_dict['Type_II_error_micro'] = self.recall_ykp()[4]
        Accuracy_dict['Type_II_error_weighted'] = self.recall_ykp()[5]
        Accuracy_dict['Precision_macro'] = self.precision_ykp()[0]
        Accuracy_dict['Precision_micro'] = self.precision_ykp()[1]
        Accuracy_dict['Precision_weighted'] = self.precision_ykp()[2]
        Accuracy_dict['Recall_macro'] = self.recall_ykp()[0]
        Accuracy_dict['Recall_micro'] = self.recall_ykp()[1]
        Accuracy_dict['Recall_weighted'] = self.recall_ykp()[2]
        Accuracy_dict['AUC_ovr_macro'] = self.auc_ykp()[0]
        Accuracy_dict['AUC_ovr_weighted'] = self.auc_ykp()[1]
        Accuracy_dict['AUC_ovo_macro'] = self.auc_ykp()[2]
        Accuracy_dict['AUC_ovo_weighted'] = self.auc_ykp()[3]
        Accuracy_dict['Jaccard_macro'] = self.jaccard_ykp()[0]
        Accuracy_dict['Jaccard_micro'] = self.jaccard_ykp()[1]
        Accuracy_dict['Jaccard_weighted'] = self.jaccard_ykp()[2]
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
    # ==== 计算多标签精度的测试 ====
    y_true = [0, 1, 1, 1, 0, 2, 2]
    y_pre_label = [0, 1, 2, 2, 0, 2, 1]
    y_pre_prob = [[0.5, 0.2, 0.3], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.2, 0.6],
                  [0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]]
    AccSet = AccuracyMeasureMultiYkp(y_true, y_pre_label, y_pre_prob)
    AccDict = AccSet.accuracy_dict_ykp()
    # 将dict转换为DataFrame
    AccDf = pd.DataFrame(AccDict, index=[0])
    print(" 多标签精度结果 \n", AccDf.T)
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
