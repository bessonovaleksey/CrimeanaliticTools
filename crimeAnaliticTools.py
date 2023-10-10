#!/usr/bin/env python
"""
Statistical methods for studying criminal activity using artificial intelligence (AI)

Copyright (c) 2023, A.A. Bessonov (bestallv@mail.ru)

Routines in this module:

ci_auc(y_true, y_pred, conf_level=0.95, n_bootstraps=1000, method='delong', sample_weight=None)
confusion_matrix_statistic(y_true, y_pred, pos_label=1, conf_level=0.95)
Dixon_Test(dataframe, q=95, info=True, autorm=False)
ensemble_combine(x,Y,func,test_size=0.3,n_splits=5,type_target='bin',progress_bar=None,verbose=True,
          param_grid=None,random_state_tts=None,random_state_KF=None,random_state_f=None)
garson_nn(model, figsize=(10,7), bar_plot=True, title=None, features_names=None,
        plotstyle='_mpl-gallery', **kwargs)
Grubbs_Test(input_data,q=95,num_outliers=3)
importance_feature(model,figsize=(10,7),title=None,y_label=None,**kwargs)
inter_input_weight(model, features_names=None)
Irvin_Test(data,q=0.05,method=1)
lekprofile_nn(model, data, xsel=None, steps=100, group_vals=np.arange(0,1.1,0.2), val_out=False,
              group_show=False, figsize=(10, 7), width=0.8, plotstyle='_mpl-gallery')
olden_nn(model, x_names=None, y_names=None, out_var=None, figsize=(10,7), bar_plot=True,
              title=None, plotstyle='_mpl-gallery', **kwargs
plot_fit_history(history, figsize=(10,7))
plot_MLP(model, left=.1, right=.9, bottom=.1, top=.9, figsize=(15,10), lab_fontsize=12, wt_fontsize=10,
             edges_color=['gray','k'], nodes_color='w', features_names=None, target_names=None, plot_fig=True,
             save_fig=False, file_name='nn_diagram.png', **kwargs)
progress(it,total,buffer=30)
Rait_Test(input_data,method=1)
regressor_combine(x, Y, func, test_size=0.3, n_splits=5, progress_bar=None, verbose=True, param_grid=None,
              random_state_tts=None, random_state_KF=None, random_state_f=None)
scale(data, center=True, scale=True)



"""
from __future__ import division, absolute_import, print_function

__all__ = ['ci_auc', 'confusion_matrix_statistic', 'Dixon_Test', 'ensemble_combine', 'garson_nn',
           'Grubbs_Test', 'importance_feature', 'inter_input_weight', 'Irvin_Test', 'lekprofile_nn',
           'olden_nn','plot_fit_history', 'plot_MLP', 'progress', 'Rait_Test', 'regressor_combine',
           'scale']

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import metrics
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import log_loss
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from clusterTools import classAgreement
from functools import reduce

def ci_auc(y_true, y_pred, conf_level=0.95, n_bootstraps=1000, method='delong', sample_weight=None):
    """
    Computes the area under the curve (AUC), AUC variance and the confidence interval (CI) of AUC.

    Parameters
    ----------
    y_true: [array] The target values (class labels) of 0 and 1.
    y_pred: [array] The predicted values.
    сonf_level : [float] The width of the confidence interval as [0,1], never in percent. By default, 0.95.
    n_bootstraps : [int] The number of bootstrap replicates for method='bootstrap'. By default, 1000.
    method : [str] the method to use, either 'delong' or 'bootstrap'.
    sample_weight : [array] Sample weights. Only for method='delong'.

    Returns
    ----------
    AUC, AUC variance, CI AUC.
    """
    if (conf_level > 1 or conf_level < 0):
        raise ValueError('conf_level must be within the interval [0,1].')
    if (method not in ['delong','bootstrap']):
        raise ValueError("Invalid method, must be 'delong' or 'bootstrap'.")
    def compute_midrank_weight(x, sample_weight):
        """
        Computes midranks.

        Parameters
        ----------
        x : [array] A 1D numpy array.
        sample_weight : [array] Sample weights.
        Returns
        ----------
        Array of midranks.
        """
        J = np.argsort(x)
        Z = x[J]
        cumulative_weight = np.cumsum(sample_weight[J])
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = cumulative_weight[i:j].mean()
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2
    assert np.array_equal(np.unique(y_true), [0, 1]), 'class labels must be of 0 and 1'
    if method=='delong':
        order = (-y_true).argsort()
        label_1_count = int(y_true.sum())
        predictions_sorted_transposed = y_pred[np.newaxis, order]
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]
        tx = np.empty([k, m])
        ty = np.empty([k, n])
        tz = np.empty([k, m + n])
        if sample_weight is None:
            for r in range(k):
                tx[r, :] = stats.rankdata(positive_examples[r, :])
                ty[r, :] = stats.rankdata(negative_examples[r, :])
                tz[r, :] = stats.rankdata(predictions_sorted_transposed[r, :])
            aucs = np.round((tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n), 4)
            v01 = (tz[:, :m] - tx[:, :]) / n
            v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m     
        else:
            ordered_sample_weight = sample_weight[order]
            for r in range(k):
                tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
                ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
                tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
            total_positive_weights = sample_weight[:m].sum()
            total_negative_weights = sample_weight[m:].sum()
            pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
            total_pair_weights = pair_weights.sum()
            aucs = np.round(((sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights), 4)
            v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
            v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights   
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = (sx / m + sy / n)
        auc_std = np.sqrt(delongcov)
        if auc_std==0:
            auc_std=0.00001
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - conf_level) / 2)
        ci = np.round(stats.norm.ppf(lower_upper_q,loc=aucs,scale=auc_std), 4)
        ci[ci > 1] = 1
        ci[ci < 0] = 0
        return aucs[0], round(delongcov,4), list(ci)
    if method=='bootstrap':
        bootstrapped_scores = []
        rng = np.random.RandomState(123)
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        aucs=round(roc_auc_score(y_true, y_pred), 4)
        ci_lower = round(sorted_scores[int(round((1-conf_level)/2, 2) * len(sorted_scores))], 4)
        ci_upper = round(sorted_scores[int((conf_level+(1-conf_level)/2) * len(sorted_scores))], 4)
        ci=[ci_lower, ci_upper]
        return aucs, ci


def confusion_matrix_statistic(y_true, y_pred, pos_label=1, conf_level=0.95):
    """
    Calculates a cross-tabulation of observed and predicted classes with associated statistics.

    Parameters
    ----------
    y_true: [array] The target values (class labels).
    y_pred: [array] The predicted values.
    pos_label : [int] The factor level that corresponds to a 'positive' result (if that
                    makes sense for your data). By default, pos_label=1.
    conf_level : [float] The width of the confidence interval as [0,1], never in percent.
                    By default, 0.95.

    Returns
    ----------
    Confusion_matrix.
    The table with elements: Accuracy, CI Accuracy, Balanced Accuracy, No Information Rate,
    p-value, Kappa, Mcnemar's test p=value, Sensitivity (Recall), Specificity, PPV, NPV,
    Miss Value, Fall-out, Prevalence, Detection Rate, Detection Prevalence, Precision, F1,
    AUC, CI AUC. For Multi-label case the table contains a statistics by classes.
    """
    if (conf_level > 1 or conf_level < 0):
        raise ValueError('conf_level must be within the interval [0,1].')
    unique, count = np.unique(y_true, return_counts=True)
    unique_, count_ = np.unique(y_pred, return_counts=True)
    classM=pd.DataFrame([count,count_],index=['Reference','Prediction'],columns=unique)
    cm=confusion_matrix(y_true, y_pred)
    ac=round(classAgreement(cm)['diag'],4)
    kappa=round(classAgreement(cm)['kappa'],4)
    res=binomtest(np.trace(cm), n=np.sum(cm), p=0.5)
    CI=res.proportion_ci(confidence_level=conf_level)
    if len(np.unique(y_true))==2:
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
        se=tp/(tp+fn)
        sp=tn/(fp+tn)
        res2=binomtest(np.trace(cm), n=np.sum(cm), p=(max(tp+fp,fn+tn)/(tp+fp+fn+tn)), alternative='greater')
        nir=max(tp+fp,fn+tn)/(tp+fp+fn+tn)
        if pos_label == 0:
            prev=(tp+fp)/(tp+fp+fn+tn)
        if pos_label == 1:
            prev=(fn+tn)/(tp+fp+fn+tn)
        ppv=(se*prev)/((se*prev)+((1-sp)*(1-prev)))
        npv=(sp*(1-prev))/(((1-se)*prev)+((sp)*(1-prev)))
        fnr=fn/(fn+tp)
        fpr=fp/(fp+tn)
        dr=tp/(tp+fp+fn+tn)
        dp=(tp+fn)/(tp+fp+fn+tn)
        bac=(se+sp)/2
        mcn = mcnemar(cm)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        f1score = f1_score(y_true, y_pred, pos_label=pos_label)
        bac=(se+sp)/2
        fpr_, tpr_, _ = metrics.roc_curve(y_true, y_pred)
        auc_=metrics.auc(fpr_, tpr_)
        ci_auc_=ci_auc(y_true, y_pred, conf_level=conf_level)[2]
        ind=['Accuracy', f'{conf_level} CI Accuracy', 'Balanced Accuracy', 'No Information Rate', 'p-value [Acc > NIR]', 'Kappa', 'Mcnemar test p-value', 'Sensitivity (Recall)', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Miss Value', 'Fall-out', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Precision', 'F1','AUC', f'{conf_level} CI AUC', 'Positive Class']
        stats=np.array([ac,[round(CI.low, 4), round(CI.high, 4)],bac,nir,res2.pvalue,kappa,mcn.pvalue,recall,sp,ppv,npv,fnr,fpr,prev,dr,dp,precision,f1score,auc_,ci_auc_,pos_label],dtype='object')
        tableStats=pd.DataFrame(stats,index=ind,columns=['Statistics'])
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print('Confusion Matrix')
        print(pd.DataFrame(cm,index=unique,columns=unique),end='\n\n')
        print('Classes Matrix')
        print(classM,end='\n\n')
        print(tableStats)
    if len(np.unique(y_true))>2:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
        se=tp/(tp+fn)
        sp=tn/(fp+tn)
        res2=binomtest(np.trace(cm), n=np.sum(cm), p=(np.max(np.sum(cm,axis=0))/np.sum(cm)), alternative='greater')
        nir=round(np.max(np.sum(cm,axis=0))/np.sum(cm),4)
        prev=[]
        for i in unique:
            prev.append(count[i]/np.sum(count))
        prev=np.array(prev)
        ppv=(se*prev)/((se*prev)+((1-sp)*(1-prev)))
        npv=(sp*(1-prev))/(((1-se)*prev)+((sp)*(1-prev)))
        fnr=fn/(fn+tp)
        fpr=fp/(fp+tn)
        dr=tp/(tp+fp+fn+tn)
        dp=(tp+fn)/(tp+fp+fn+tn)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        f1score = f1_score(y_true, y_pred, pos_label=pos_label)
        mcn = mcnemar(cm)
        mcc=matthews_corrcoef(y_true, tr_res)
        bac=(se+sp)/2
        roc_auc=[]
        for i in unique:
            fpr_, tpr_, _ = metrics.roc_curve(y_true==i, y_pred==i)
            roc_auc.append(metrics.auc(fpr_, tpr_))
        colN=[]
        for i in unique:
            colN.append(str(f'Class: {i}'))
        ind=['Sensitivity (Recall)', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Miss Value', 'Fall-out', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Precision', 'F1','Balanced Accuracy','AUC']
        stats=np.array([recall,sp,ppv,npv,fnr,fpr,prev,dr,dp,precision,f1score,bac,roc_auc])
        tableStats=pd.DataFrame(stats,index=ind,columns=colN)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print('Confusion Matrix')
        print(pd.DataFrame(cm,index=unique,columns=unique),end='\n\n')
        print('Classes Matrix')
        print(classM,end='\n\n')
        print(f'\nAccuracy : {ac}')
        print(f'{conf_level} CI Accuracy : {[round(CI.low, 4), round(CI.high, 4)]}')
        print(f'No Information Rate : {nir}')
        print(f'p-value [Acc > NIR] : {res2.pvalue}')
        print(f'Kappa : {kappa}')
        print(f'Mcnemars test p-value : {round(mcn.pvalue, 4)}')
        print(f'Matthews test : {round(mcc, 4)}')
        print('\nStatistics by Class:')
        print(tableStats)


def Dixon_Test(dataframe, q=95, info=True, autorm=False):
    '''
    Dixon test to identify outliers
    
    Dixon Q Test algorithm.vRemove all outliers from the vector. Returns cleared dataframe is autorm is True.
    
    Parameters
    ----------
    dataframe : [DataFrame, array] dataframe or vector of data to be verified for outliers.
    q : [int] the level of significance of the test: 90, 95 (default), 99.
    info : [bool] if the True (default) will be dataframe with emissions is printed.
    autorm : [bool] if the True (default) will be outliers delete before the next test and input data frame will be return without identified  outliers.
    
    Returns
    ----------
    Data frame with Detected outliers(info=True) and the input date frame without identified  outliers
    '''
    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
        0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
        0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
        0.277, 0.273, 0.269, 0.266, 0.263, 0.26]
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
        0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
        0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
        0.308, 0.305, 0.301, 0.29]
    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
        0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
        0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
        0.384, 0.38, 0.376, 0.372]
    Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
    Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
    Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}
    if type(dataframe) == pd.Series:
        dataframe=pd.DataFrame(dataframe)
    def dixon_test(data, q_dict=Q95):
        sdata = sorted(data)
        Q_mindiff, Q_maxdiff = (0,0), (0,0)
        if len(data)>2 and len(data)<8:
            Q_min = (sdata[1] - sdata[0])/(sdata[-1]-sdata[0])
        if len(data)>7 and len(data)<11:
            Q_min = (sdata[1] - sdata[0])/(sdata[-2]-sdata[0])
        if len(data)>10 and len(data)<14:
            Q_min = (sdata[2] - sdata[0])/(sdata[-2]-sdata[0])
        if len(data)>13:
            Q_min = (sdata[2] - sdata[0])/(sdata[-3]-sdata[0])
        if len(data)<=28:
            Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])
        if len(data)>28:
            Q_mindiff = (Q_min - q_dict[28], sdata[0])
        if len(data)>2 and len(data)<8:
            Q_max = (sdata[-1] - sdata[-2])/(sdata[-1]-sdata[0])
        if len(data)>7 and len(data)<11:
            Q_max = (sdata[-1] - sdata[-2])/(sdata[-1]-sdata[1])
        if len(data)>10 and len(data)<14:
            Q_max = (sdata[-1] - sdata[-3])/(sdata[-1]-sdata[1])
        if len(data)>13:
            Q_max = (sdata[-1] - sdata[-3])/(sdata[-1]-sdata[2])
        if len(data)<=28:
            Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])
        if len(data)>28:
            Q_maxdiff = (Q_max - q_dict[28], sdata[-1])
        if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
            outliers = []
        elif Q_mindiff[0] == Q_maxdiff[0]:
            outliers = [Q_mindiff[1], Q_maxdiff[1]]
        elif Q_mindiff[0] > Q_maxdiff[0]:
            outliers = [Q_mindiff[1]]
        elif Q_mindiff[0] < 0:
            outliers = [Q_mindiff[1], Q_maxdiff[1]]
        else:
            outliers = [Q_maxdiff[1]]
        return outliers
    for column in dataframe:
        vector = np.array(dataframe[column])
        if q == 90:
            outliers = dixon_test(vector, q_dict=Q90)
        if q == 95:
            outliers = dixon_test(vector, q_dict=Q95)
        if q == 99:
            outliers = dixon_test(vector, q_dict=Q99)
        if len(outliers) > 0:
            for elem in outliers:
                condition = dataframe[column] == elem
                out = dataframe[column].index[condition]
                if info is True:
                    print(f'Detected outlier: \n{dataframe.loc[[out[0]]]}')
                if autorm is True:
                    dataframe.drop(index=out, inplace = True)
    return dataframe


def garson_nn(model, figsize=(10,7), bar_plot=True, title=None, features_names=None, plotstyle='_mpl-gallery', **kwargs):
    """
    Variable importance using Garson's algorithm.

    Parameters
    ----------
    model : the sklearn.neural_network estimator or a Keras model.
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                    (default figsize=(10,7)).
    bar_plot : [bool] If True, return plot of the relative importance of features. Otherwise, return 
                    the date frame with the values of the relative importance of features.
                    By default, True.
    title : [str] plot main title. By, default None.
    features_names : [str] the label text of features from attribute of the model feature_names_in_ 
                    or a custom names. By default, None.
    plotstyle : [str] stylesheets from Matplotlib for plot, default '_mpl-gallery'.
    **kwargs : other arguments for matplotlib.pyplot.bar.

    Returns
    ----------
    A bar plot of the relative importance of features or date frame with the values
    of the relative importance of features.
    """
    if 'keras' in str(type(model)):
        model_coefs = []
        for layer in model.layers: model_coefs.append(layer.get_weights())
        model_coefs = reduce(lambda x, y: x+y, model_coefs)
        model_coefs = [el for i, el in enumerate(model_coefs) if i%2==0]
        num_outputs = model.layers[-1].get_config().get('units')
        max_i = len(model.layers)+1
        mod_str = str(type(model))
        name_model = 'Variable importance for {}'.format(mod_str[mod_str.find(' ')+1 : mod_str.find('>')])
        n_features = model.layers[0].get_config().get('batch_input_shape')[1]
    if 'sklearn' in str(type(model)):
        num_outputs = model.n_outputs_
        model_coefs = model.coefs_
        max_i = model.n_layers_
        name_model = 'Variable importance for {}'.format(str(model).split("(")[0])
        n_features = model.n_features_in_
    if num_outputs > 1:
        raise ValueError('Garson only applies to neural networks with one output node')
    if max_i > 3:
        raise ValueError('Garsons algorithm not applicable for multiple hidden layers')    
    if title is None:
        title = name_model
    else:
        title = title
    if features_names is None:
        if hasattr(model,'feature_names_in_'):
            features_names = model.feature_names_in_
        else:
            features_names = []
            for i in range(1,n_features+1):
                features_names.append(r'X_{}'.format(i))
            features_names = np.array(features_names)
    else:
        features_names = features_names
    sum_in=np.multiply(abs(model_coefs[0].T), abs(model_coefs[1]))
    sum_in=sum_in/np.sum(sum_in,axis=1)[:,None]
    sum_in=np.sum(sum_in,axis=0)
    rel_imp=sum_in/np.sum(sum_in)
    index_sorted=np.flipud(np.argsort(rel_imp))
    pos = np.arange(index_sorted.shape[0]) + 0.5
    if bar_plot is False:
        out=pd.DataFrame({'Relative importance':rel_imp}, index = feature_names)
        return out
    if bar_plot is True:
        colors = plt.cm.Blues(np.linspace(0.9, 0.3, len(rel_imp)))
        plt.figure(figsize=figsize, facecolor='w')
        plt.style.use(plotstyle)
        plt.bar(pos, rel_imp[index_sorted], align='center', color=colors, **kwargs)
        plt.xticks(pos, features_names[index_sorted], rotation=90)
        plt.ylabel('Importance')
        plt.title(title)
        plt.show()


def Grubbs_Test(input_data,q=95,num_outliers=3):
    '''
    Grabbs test to identify outliers

    Parameters
    ----------
    input_data : [Series] vector of data to be verified for outliers.
    q : [int] the level of significance of the test: 95 (default) or 99.
    num_outliers : [int] the number of identified outliers (default num_outliers=3).

    Returns
    ----------
    Value is an outlier or not an outlier, Grubbs Statistics Value, Grubbs Critical Value
    '''
    from scipy import stats
    q95 = [1.153, 1.463, 1.672, 1.822, 1.938, 2.032, 2.11, 
        2.176, 2.234, 2.285, 2.331, 2.371, 2.409, 2.443, 2.475, 
        2.504, 2.532, 2.557, 2.58, 2.603, 2.624, 2.644, 2.663, 
        2.681, 2.698, 2.714, 2.73, 2.745, 2.759, 2.773, 2.786, 
        2.799, 2.811, 2.823, 2.835, 2.846, 2.857, 2.866, 2.877, 
        2.887, 2.896, 2.905, 2.914, 2.923, 2.931, 2.94, 2.948, 
        2.956, 2.964, 2.971, 2.973, 2.986, 2.992, 3.0, 3.006, 
        3.013, 3.019, 3.035, 3.032, 3.037, 3.044, 3.049, 3.055, 
        3.061, 3.066, 3.071, 3.076, 3.082, 3.087, 3.092, 3.098, 
        3.102, 3.107, 3.111, 3.117, 3.121, 3.125, 3.13, 3.134, 
        3.139, 3.143, 3.147, 3.151, 3.155, 3.16, 3.163, 3.167, 
        3.171, 3.174, 3.179, 3.182, 3.186, 3.189, 3.193, 3.196, 
        3.201, 3.204, 3.207, 3.21, 3.214, 3.217, 3.22, 3.224, 
        3.227, 3.23, 3.233, 3.236, 3.239, 3.242, 3.245, 3.248, 
        3.251, 3.254, 3.257, 3.259, 3.262, 3.265, 3.267, 3.27, 
        3.274, 3.276, 3.279, 3.281, 3.284, 3.286, 3.289, 3.291, 
        3.294, 3.296, 3.298, 3.302, 3.304, 3.306, 3.309, 3.311, 
        3.313, 3.315,3.318, 3.32, 3.322, 3.324, 3.326, 3.328, 
        3.331, 3.334]  
    q99 = [1.155, 1.492, 1.749, 1.944, 2.097, 2.221, 2.323, 2.41, 
        2.485, 2.55, 2.607, 2.659, 2.705, 2.747, 2.785, 2.821, 
        2.854, 2.884, 2.912, 2.939, 2.963, 2.987, 3.009, 3.029, 
        3.049, 3.068, 3.085, 3.103, 3.119, 3.135, 3.15, 3.164, 
        3.178, 3.191, 3.204, 3.216, 3.228, 3.24, 3.251, 3.261, 
        3.271, 3.282, 3.292, 3.302, 3.31, 3.319, 3.329, 3.336,
        3.345, 3.353, 3.361, 3.368, 3.376, 3.383, 3.391, 3.397, 
        3.405, 3.411, 3.418, 3.424, 3.43, 3.437, 3.442, 3.449, 
        3.454, 3.46, 3.466, 3.471, 3.476, 3.482, 3.487, 3.492, 
        3.496, 3.502, 3.507, 3.511, 3.516, 3.521, 3.525, 3.529, 
        3.534, 3.539, 3.543, 3.547, 3.551, 3.555, 3.559, 3.563, 
        3.567, 3.57, 3.575, 3.579, 3.582, 3.586, 3.589, 3.593, 
        3.597, 3.6, 3.603, 3.607, 3.61, 3.614, 3.617, 3.62, 
        3.623, 3.626, 3.629, 3.632, 3.636, 3.639, 3.642, 3.645, 
        3.647, 3.65, 3.653, 3.656, 3.659, 3.662, 3.665, 3.667, 
        3.67, 3.672, 3.675, 3.677, 3.68, 3.683, 3.686, 3.688,
        3.69, 3.693, 3.695, 3.697, 3.7, 3.702, 3.704, 3.707, 
        3.71, 3.712, 3.714, 3.716, 3.719, 3.721, 3.723, 3.725, 
        3.727]
    Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
    Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}
    def grubbs_stat(x):
        std_dev=np.std(x)
        avg_x=np.mean(x)
        val_avg=abs(x-avg_x)
        max_deviations=max(val_avg)
        max_ind=np.argmax(val_avg)
        Gcal=max_deviations/std_dev
        return Gcal, max_ind 
    def interpretation_G_values(Gs, Gc, df, max_index):
        if Gs>Gc:
            print(f"{df[max_index]} - это выброс, G=%.4f, Gt=%.3f" % (Gs,Gc))
        else:
            print(f"{df[max_index]} - это не выброс, G=%.4f, Gt=%.3f" % (Gs,Gc))
    for iteration in range(num_outliers):
        if q == 95:
            q_dict=Q95
        if q == 99:
            q_dict=Q99
        if len(input_data)<=144:
            Gcrit = q_dict[len(input_data)]
        if len(input_data)>144:
            Gcrit = q_dict[144]
        input_data=np.array(input_data)
        Gstat,max_index=grubbs_stat(input_data)
        interpretation_G_values(Gstat,Gcrit,input_data,max_index)
        input_data=np.delete(input_data,max_index)


def ensemble_combine(x,Y,func,test_size=0.3,n_splits=5,type_target='bin',progress_bar=None,verbose=True,param_grid=None,random_state_tts=None,random_state_KF=None,random_state_f=None):
    """
    Search for the best combination of features in data using ensemble methods.

    Parameters
    ----------
    x : [DataFrame] The training input samples for searching for the best combination
                of features for use in ensemble methods.
    Y : [array] The target values (class labels) as integers or strings.
    func : a function from the module sklearn.ensemble or sklearn.linear_model.
    test_size : [float or int] If float, should be between 0.0 and 1.0 and represent 
                the proportion of the dataset to include in the test split. If int, 
                represents the absolute number of test samples. By default, 0.3.
    n_splits : [int] Number of folds. Must be at least 2. By defailt, 5.
    type_target : [str] The type the classification: if 'bin' (default) it is binary classification
                (where the labels are [0, 1] or [-1, 1]), if 'multi' it is multiclass
                classification (where the labels are [0, …, K-1]).
    progress_bar : [bool] If True to show iterations of algorithm. By default, None.
    verbose : [bool] If True to show number of combinations of features in the data.
                By default, True.
    param_grid : [str] The function's ('func') parameter that changes during the operation of the
                algorithm. By default, None.
    random_state_tts : Controls the shuffling applied to the data before applying the split
                for sklearn.model_selection.train_test_split. By defailt, None.
    random_state_KF : Controls affects the ordering of the indices, which controls the randomness 
                of each fold for sklearn.model_selection.KFold. By defailt, None.
    random_state_f : Controls the randomness of the estimator, which indicated in the func.
                By defailt, None.

    Returns
    ----------
    The table with columns:
    'Parameter' : parameter values of function (if param_grid is not None);
    'ACC train' : accuracity of training the model;
    'ACC test'  : accuracity of testing the model;
    'AUC'       : Area Under the Curve (AUC) when testing the model;
    'loss'      : loss when testing the model;
    'Precision' : the precision;
    'Recall'    : the recall (sensitivity);
    'F1'        : F-measure;
    'features'  : combination of features in data.
    """
    from sklearn.metrics import auc
    from itertools import combinations
    from sklearn.model_selection import KFold, cross_val_score
    nc=list(x.columns)
    y=[]
    for i in range(2,len(nc)+1):
        y1=list(combinations(nc, i))
        y.extend(y1)
    if verbose is True:
        print('%d combinations of features in the data' % (len(y)))
    if param_grid==None:
        col_names =  ['ACC train', 'ACC test', 'AUC', 'loss', 'Precision', 'Recall', 'F1', 'features']
    else:
        col_names =  ['Parameter ', 'ACC train', 'ACC test', 'AUC', 'loss', 'Precision', 'Recall', 'F1', 'features']
        grid=ParameterGrid(param_grid)
    my_df  = pd.DataFrame(columns = col_names)
    if verbose is True and param_grid != None:
        print('%d combinations will be checked in total' % (len(y)*len(grid)))
    for j in range(0,len(y)):
        X=x[list(y[j])]
        D=pd.concat([X,Y],axis=1)
        if type_target=='bin':
            train,test = train_test_split(D,test_size=test_size,random_state=random_state_tts)
            X_train, Y_train = train[train.columns[:-1]], train[train.columns[-1]]
            X_test, Y_test = test[test.columns[:-1]], test[test.columns[-1]]
        if type_target=='multi':
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=(1-test_size),random_state=random_state_tts,stratify=Y)
        if param_grid != None:
            for param in grid:
                classifier = func(random_state=random_state_f,**param).fit(X_train, Y_train)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_KF)
                results = cross_val_score(classifier, X_train, Y_train, cv=kf, scoring="roc_auc")
                Y_pred = classifier.predict(X_test)
                false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
                roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
                nn = list(X.columns)
                res = round(classifier.score(X_test, Y_test), 3)
                acc = round(sum(results)/len(results), 3)
                loss = round(log_loss(Y_test, Y_pred), 2)
                precision = round(precision_score(Y_test, Y_pred) ,3)
                recall = round(recall_score(Y_test, Y_pred), 3)
                f1score = round(f1_score(Y_test, Y_pred), 3)
                col_names =  ['Parameter ', 'ACC train', 'ACC test', 'AUC', 'loss', 'Precision', 'Recall', 'F1', 'features']
                p_data = [(''.join('{}:{} '.format(key, val) for key, val in param.items()))]
                df2 = pd.DataFrame({'Parameter ': p_data, 'ACC train': acc, 'ACC test': res, 'AUC': roc_auc, 'loss': loss, 'Precision': precision, 'Recall': recall, 'F1' : f1score, 'features': [nn]})
                my_df = pd.concat([my_df, df2],ignore_index=True)
                my_df = my_df.reset_index(drop=True)
                my_df = my_df.rename(index = lambda x: x + 1)
                if progress_bar is True:
                    progress(my_df.shape[0]+1,len(y)*len(grid))
        else:
            classifier = func(random_state=random_state_f).fit(X_train, Y_train)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_KF)
            results = cross_val_score(classifier, X_train, Y_train, cv=kf, scoring="roc_auc")
            Y_pred = classifier.predict(X_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
            roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
            nn = list(X.columns)
            res = round(classifier.score(X_test, Y_test), 3)
            acc = round(sum(results)/len(results), 3)
            loss = round(log_loss(Y_test, Y_pred), 2)
            precision = round(precision_score(Y_test, Y_pred) ,3)
            recall = round(recall_score(Y_test, Y_pred), 3)
            f1score = round(f1_score(Y_test, Y_pred), 3)
            col_names =  ['ACC train', 'ACC test', 'AUC', 'loss', 'Precision', 'Recall', 'F1', 'features']
            df2 = pd.DataFrame({'ACC train': acc, 'ACC test': res, 'AUC': roc_auc, 'loss': loss, 'Precision': precision, 'Recall': recall, 'F1' : f1score, 'features': [nn]})
            my_df = pd.concat([my_df, df2],ignore_index=True)
            my_df = my_df.reset_index(drop=True)
            my_df = my_df.rename(index = lambda x: x + 1)
            if progress_bar is True:
                progress(my_df.shape[0]+1,len(y))
    return my_df


def importance_feature(model,figsize=(10,7),title=None,y_label=None,**kwargs):
    """
    Assessment of the relative importance of features for estimator.

    Parameters
    ----------
    model : the scikit-learn estimator, which has an attribute feature_importances_.
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                    (default figsize=(12,7)).
    title : [str] plot main title. By, default None.
    y_label : [str] the label text of features from attribute of the model feature_names_in_.
                    By, default None.
    **kwargs : other arguments for matplotlib.pyplot.bar.

    Returns
    ----------
    A bar plot of the relative importance of features.
    """
    feature_importances = model.feature_importances_
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    index_sorted=np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0]) + 0.5
    feature_names = model.feature_names_in_
    if title is None:
        title='Assessment of the importance of features for {}'.format(model)
    else:
        title=title
    if y_label is None:
        y_label='Relative Importance'
    else:
        y_label=y_label
    plt.figure(figsize=figsize, facecolor='w')
    plt.bar(pos, feature_importances[index_sorted], align='center',**kwargs)
    plt.xticks(pos, feature_names[index_sorted], rotation=90)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def inter_input_weight(model, features_names=None):
    """
    The number of positiv and negative weights of input neurons neural net are returned.

    Parameters
    ----------
    model : the sklearn.neural_network estimator or Keras model.
    features_names : [str] the label text of features from attribute of the model
                    feature_names_in_ or a custom names. By default, None.

    Returns
    ----------
    data frame of the number of positiv and negative weights of input neurons
    neural net for explanator variables.
    """
    if 'keras' in str(type(model)):
        model_coefs = []
        for layer in model.layers: model_coefs.append(layer.get_weights())
        model_coefs = reduce(lambda x, y: x+y, model_coefs)
        model_coefs = [el for i, el in enumerate(model_coefs) if i%2==0]
        n_features = model.layers[0].get_config().get('batch_input_shape')[1]
    if 'sklearn' in str(type(model)):
        model_coefs = model.coefs_
        n_features = model.n_features_in_
    if features_names is None:
        if hasattr(model,'feature_names_in_'):
            features_names = model.feature_names_in_
        else:
            features_names = []
            for i in range(1,n_features+1):
                features_names.append(r'X_{}'.format(i))
            features_names = np.array(features_names)
    else:
        features_names = features_names

    clf_arr = np.zeros((len(model_coefs[0]), 4))
    for i in range(len(model_coefs[0])):
        clf_arr[i,[0]] = sum(i > 0 for i in model_coefs[0][i])
        clf_arr[i,[1]] = sum(i < 0 for i in model_coefs[0][i])
        clf_arr[i,[2]] = np.round(clf_arr[i,[0]]/len(model_coefs[0][0])*100,1)
        clf_arr[i,[3]] = np.round(clf_arr[i,[1]]/len(model_coefs[0][0])*100,1)
    res=pd.DataFrame(clf_arr,columns=['positive','negative','pos %','neg %'],index=features_names)
    res=res.astype({'positive': 'Int64','negative': 'Int64'})
    return res


def Irvin_Test(data,q=0.05,method=1):
    """
    Irvin test to identify outliers

    Parameters
    ----------
    data : [Series] vector of data to be verified for outliers.
    q : [int] the level of significance of the test: 0.1, 0.05 (default) or 0.01.
    method : Irwin test method: 1 - for maximum and minimum values (default), 2 - for the full data sample.

    Returns
    ----------
    Value is an outlier, Statistics of the of Irwin's criterion (L), critical significance for the Irwin's criterion (Lcrit) or indication of the absence of outliers
    """
    sdata = sorted(data)
    n=len(sdata)
    if q==0.1:
        qkrit=-132.78*n**(-3)+224.24*n**(-2.5)-165.27*n**(-2)+68.614*n**(-1.5)-16.109*n**(-1)+3.693*n**(-0.5)+0.549
    if q==0.05:
        qkrit=-229.21*n**(-3)+422.39*n**(-2.5)-320.96*n**(-2)+124.594*n**(-1.5)-26.15*n**(-1)+4.799*n**(-0.5)+0.7029
    if q==0.01:
        qkrit=-205.06*n**(-3)+424.26*n**(-2.5)-352.483*n**(-2)+143.747*n**(-1.5)-33.401*n**(-1)+6.381*n**(-0.5)+1.049
    sdata = sorted(data)
    if method==1:
        Imin=abs((sdata[1]-sdata[0])/np.std(sdata))
        Imax=abs((sdata[-1]-sdata[-2])/np.std(sdata))
        if Imax>=qkrit:
            print(f"{sdata[-1]} - это выброс, L=%.4f, Lcrit=%.4f" % (Imax,qkrit))
        if Imin>=qkrit:
            print(f"{sdata[0]} - это выброс, L=%.4f, Lcrit=%.4f" % (Imin,qkrit))
        if Imax<qkrit and Imin<qkrit:
            print('Выбросов нет')
    if method==2:
        def irvin_test(x,data):
            median=np.median(data)
            if x < median:
                Icr=abs((data[data.index(x)+1]-data[data.index(x)])/np.std(data))
                return Icr
            if x > median:
                Icr=abs((data[data.index(x)]-data[data.index(x)-1])/np.std(data))
                return Icr
            if x == median:
                Icr=0
                return Icr
        out_df=[irvin_test(i,sdata) for i in sdata]
        if all([x<qkrit for x in out_df]):
            print('Выбросов нет')
        for num in out_df:
            if num>=qkrit:
                ind=out_df.index(num)
                outlier=sdata[ind]
                print(f"{outlier} - это выброс, L=%.4f, Lcrit=%.4f" % (num,qkrit))


def scale(data, center=True, scale=True):
    """
    This is a generic function which centers and scales the columns
    of a dataframe or array.

    Parameters
    ----------
    data :   [DataFrame, array] dataframe or array for centering, scaling.
    center : [bool] If True (default) then centering is done by subtracting
              the column means of data from their corresponding columns, 
              and if center=False, no centering is done.
    scale :  [bool] If True (default) then scentered columns of the 
              dataframe/array is divided by the root mean square. 
              If scale=False, no scaling is done.

    Returns
    ----------
    Dateframe or array which scaled and/or centered and mean values by columns.

    """
    x = data.copy()
    if center:
        x -= np.mean(x,axis=0)
        xsc = np.mean(x,axis=0)
        return x, xsc
    if scale and center:
        x /= np.std(x,axis=0)
        xstd = np.std(x,axis=0)
        return x, xstd
    elif scale:
        x /= np.sqrt(np.sum(np.power(x,2),axis=0)/(x.shape[0]-1))
    return x
    

def lekprofile_nn(model, data, xsel=None, steps=100, group_vals=np.arange(0,1.1,0.2), val_out=False, group_show=False, figsize=(10, 7), width=0.8, plotstyle='_mpl-gallery'):
    """
    Sensitivity analysis using Lek's profile method for input variables.

    Conduct a sensitivity analysis of model responses in a neural network to input 
    variables using Lek's profile method.


    Parameters
    ----------
    model : the sklearn.neural_network estimator or a Keras model.
    data : [dataframe] the dataframe of explanatory variables used to create the input model.
    xsel : [list of str] list of names of explanatory variables to plot, defaults to all.
    steps : [int] numeric value indicating number of observations to evaluate for each explanatory
                    variable from minimum to maximum value, default 100.
    group_vals : [int or array of float] array with values from 0-1 indicating quantile values
                    at which to hold other explanatory variables constant or a single value indicating
                    number of clusters to define grouping scheme. By default, np.arange(0,1.1,0.2).
    val_out : [bool] if True, actual sensitivity values are returned rather than a plot. By default, False.
    group_show : [bool] if True, a barplot is returned that shows the values at which explanatory 
                    variables were held constant while not being evaluated. By default, False.
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                    (default figsize=(10,7)).
    width : [float] The width(s) of the bars if a barplot is returned, default 0.8.
    plotstyle : [str] stylesheets from Matplotlib for plot, default '_mpl-gallery'.


    Returns
    ----------
    if val_out=False and group_show=False, plot of groups unevaluated explanatory variables
                at quantiles or groups by cluster means are returned.
    if val_out=True, a two-element tuple with a data frame in long form showing the predicted
                responses at different values of the explanatory variables and the grouping scheme that
                was used to hold unevaluated variables constant are returned.
    if group_show=True, a stacked bar plot for each group with heights within each bar 
                proportional to the constant values is returned.
    """
    if isinstance(data, np.ndarray):
        raise ValueError('data must be a dataframe')
    if isinstance(data, pd.Series):
        raise ValueError('Lek profile requires greater than one input variable')
    if xsel is None:
        xsel = data.columns.to_list()
    else:
        xsel = xsel
        if len(xsel) == 1:
            raise ValueError('Lek profile requires greater than one input variable')
    indexes = data.columns.to_list()
    ind = list(x for x in range(len(indexes)) if indexes[x] in xsel)
    if isinstance(group_vals, (np.ndarray, list)) or group_vals == 1:
        grps_=np.quantile(data, group_vals, axis=0)
    else:
        if isinstance(group_vals, (np.ndarray, list)):
            raise ValueError('group_vals must be a single value')
        grps_=KMeans(n_clusters=group_vals, random_state=123,n_init='auto').fit(data).cluster_centers_
    if len(grps_.shape) == 1:
        grps=pd.DataFrame(columns=data.columns)
        grps.loc[0] = list(grps_)
    else:
        grps=pd.DataFrame(grps_,columns=data.columns)
    if group_show == True and val_out == False:
        grps.plot.bar(rot=0, figsize=figsize, legend=True, xlabel='Groups', \
                            ylabel='Constant values', width=width)
    if isinstance(group_vals, (np.ndarray, list)):
        g_ind=list(i+'%' for i in ((group_vals*100).astype('int')).astype('str'))
        grps = grps.set_axis(g_ind)
    lek_vals=[]
    for var_sel in data.columns:
        chngs=np.linspace(data[var_sel].min(), data[var_sel].max(), num=steps)
        const = grps.drop(var_sel,axis = 1)
        res=[]
        for i in range(const.shape[0]):
            topred=np.repeat(const.iloc[[i]].values, steps, axis=0)
            topred=np.insert(topred, 0, chngs, axis=1)
            topred=pd.DataFrame(topred,columns=data.columns)
            if 'keras' in str(type(model)):
                preds=model.predict(topred,verbose=0)
                preds=[1 if x>0.5 else 0 for x in preds]
            if 'sklearn' in str(type(model)):
                preds=model.predict(topred)
            preds=np.vstack([preds, chngs]).T
            res.append(preds)
        lek_vals.append(res)

    lek_vals = [lek_vals[x] for x in ind]

    if val_out == False and group_show == False:
        fig = plt.figure(figsize=figsize, facecolor='white')
        plt.style.use(plotstyle)
        axes = fig.subplots(nrows=1, ncols=len(lek_vals))
        for j in range(len(lek_vals)):
            for i in range(len(lek_vals[0])):
                axes[j].plot(lek_vals[j][i][:,[1]],lek_vals[j][i][:,[0]],label=i)
            axes[j].set_title(xsel[j])
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        labels=list(np.unique(labels))
        fig.legend(lines, labels,loc = 'right', title='Groups')
        fig.text(0.5, 0.05, 'Explanatory', ha='center', va='center')
        fig.text(0.07, 0.5, 'Response', ha='center', va='center', rotation='vertical')
        plt.show()

    if val_out == True:
        ful_list=[]
        for k in range(len(lek_vals)):
            M = len(lek_vals[0])
            N = len(lek_vals[0][0])
            gropus=[x//N for x in range((M)*N)]
            arr=np.array([j for i in lek_vals[k] for j in i])
            arr=np.insert(arr,2,gropus,axis=1)
            ful_list.append(arr)
        ful_arr=np.array([j for i in ful_list for j in i])
        lengths = sum([len(sublist) for sublist in lek_vals[0]])
        exp_name=np.repeat(xsel,lengths)
        out=pd.DataFrame(ful_arr,columns=['Response','Explanatory','Groups'])
        out['exp_name']=exp_name
        out = out.astype({'Response': 'int','Groups': 'int'})
        return out, grps


def olden_nn(model, x_names=None, y_names=None, out_var=None, figsize=(10,7), bar_plot=True, title=None, plotstyle='_mpl-gallery', **kwargs):
    """
    Relative importance of input variables in neural networks as the sum of the product of raw input-hidden,
    hidden-output connection weights.

    Parameters
    ----------
    model : the sklearn.neural_network estimator or a Keras model.
    x_names : [array of str or int] input variable names, obtained from the model object.
                    By default, None.
    y_names : [list or array of str or int] response variable names, obtained from the model object.
                    By default, None.
    out_var : [str or int] indicating the response variable in the neural network object
                    to be evaluated. Only one input is allowed for models with more than one response.
                    Names must be of the form 'Y1', 'Y2', etc or 0, 1, 2, etc. By default, None.
    figsize : [int, int] a method used to change the dimension of plot window, width, height in inches 
                    (default figsize=(10,7)).
    bar_plot : [bool] If True, return plot of the relative importance of features. Otherwise, return 
                    the date frame with the values of the relative importance of features.
                    By default, True.
    title : [str] plot main title. By default, None.
    plotstyle : [str] stylesheets from Matplotlib for plot, default '_mpl-gallery'. 
    **kwargs : other arguments for matplotlib.pyplot.bar.

    Returns
    ----------
    A bar plot of the relative importance of features or date frame with the values
    of the relative importance of features.
    """
    err_1 = 'Results for this response variable cannot be returned, use out_var argument to change'
    err_2 = 'y_names must be specified for the Keras model'
    if 'keras' in str(type(model)):
        if y_names is None:
            raise ValueError(err_2)
        else:
            y_names=y_names
        model_coefs = []
        for layer in model.layers: model_coefs.append(layer.get_weights())
        model_coefs = reduce(lambda x, y: x+y, model_coefs)
        model_coefs = [el for i, el in enumerate(model_coefs) if i%2==0]
        num_outputs = model.layers[-1].get_config().get('units')
        max_i = len(model.layers)-1
        mod_str = str(type(model))
        name_model = 'Variable importance for {}'.format(mod_str[mod_str.find(' ')+1 : mod_str.find('>')])
        n_features = model.layers[0].get_config().get('batch_input_shape')[1]
    if 'sklearn' in str(type(model)):
        num_outputs = model.n_outputs_
        model_coefs = model.coefs_
        if y_names is None:
            y_names=model.classes_
        else:
            y_names=y_names
        max_i = model.n_layers_-2
        name_model = 'Variable importance for {}'.format(str(model).split("(")[0])
        n_features = model.n_features_in_
    if out_var is None:
        out_var=y_names[0]
        out_var=list(y_names).index(out_var)
        out_var_=out_var
    else:
        if out_var not in y_names:
            raise ValueError('out_var must match one: {}'.format(list(y_names)))
        if type(out_var) is int:
            if out_var >= num_outputs:
                raise ValueError(err_1)
        out_var_=out_var
        out_var=list(y_names).index(out_var)
    if out_var >= num_outputs:
        raise ValueError(err_1)
    if x_names is None:
        if hasattr(model,'feature_names_in_'):
            x_names=model.feature_names_in_
        else:
            x_names=[]
            for i in range(1,n_features+1):
                x_names.append(r'X_{}'.format(i))
            x_names=np.array(x_names)
    else:
        x_names = x_names
    if title is None:
        title = name_model
    else:
        title = title
    inp_hid = model_coefs[0:max_i]
    hid_out = model_coefs[max_i][:,[out_var]]
    sum_in=np.matmul(inp_hid[max_i-1],hid_out)
    if max_i != 1:
        for i in range(max_i-2,-1,-1):
            sum_in=np.matmul(inp_hid[i],sum_in)
            importance = sum_in
    else:
        importance = sum_in
    importance=importance.ravel()
    if bar_plot is False:
        out=pd.DataFrame({'Relative importance':importance}, index = x_names)
        return out
    if bar_plot is True:
        index_sorted=np.flipud(np.argsort(importance))
        pos = np.arange(index_sorted.shape[0]) + 0.5
        colors = plt.cm.Blues(np.linspace(0.9, 0.3, len(importance)))
        plt.figure(figsize=figsize, facecolor='w')
        plt.style.use(plotstyle)
        plt.bar(pos, importance[index_sorted], align='center', color=colors, **kwargs)
        plt.xticks(pos, x_names[index_sorted], rotation=90)
        plt.ylabel('Importance for class: {}'.format(str(out_var_)))
        plt.title(title)
        plt.show()


def plot_fit_history(history, figsize=(10,7)):
    """
    Draw of plots the loss and accuracy of the model over the training and validation
    data during training.
    
    Parameters
    ----------
    history : A history object reterned by function fit.
    figsize : [int, int] a method used to change the dimension of plot window, width,
                height in inches (default figsize=(10,7)).

    Returns
    ----------
    Plots the loss and accuracy of the model.
    """
    fig, axs = plt.subplots(2, figsize=figsize, sharey=True)
    fig.set_facecolor('w')
    plt.style.use('ggplot')
    history_list=list(history.history.keys())
    # Plot training & validation loss values
    axs[0].plot(history.history[history_list[0]], 'o-r', label='Training Loss')
    if len(history_list) > 2:
        axs[0].plot(history.history[history_list[2]], 'o-b', label='Validation Loss')
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel(history_list[0])
    axs[0].set_xlabel('epoch')
    axs[0].legend(loc='best')
    axs[0].grid(which = "major", linewidth = 1)
    axs[0].grid(which = "minor", linewidth = 0.5)
    axs[0].minorticks_on()
    axs[0].set_xlim([-.3, len(history.history[history_list[0]])])
    # Plot training & validation accuracy values
    axs[1].plot(history.history[history_list[1]], 'o-r', label='Training Accuracy')
    if len(history_list) > 2:
        axs[1].plot(history.history[history_list[3]], 'o-b', label='Validation Accuracy')
    axs[1].set_title('Model Accuracy')
    axs[1].set_ylabel(history_list[1])
    axs[1].set_xlabel('epoch')
    axs[1].legend(loc='best')
    axs[1].grid(which = "major", linewidth = 1)
    axs[1].grid(which = "minor", linewidth = 0.5)
    axs[1].minorticks_on()
    axs[1].set_xlim([-.3, len(history.history[history_list[0]])])
    plt.tight_layout()
    plt.show()


def plot_MLP(model, left=.1, right=.9, bottom=.1, top=.9, figsize=(15,10), lab_fontsize=12, wt_fontsize=10, edges_color=['gray','k'], nodes_color='w', features_names=None, target_names=None, plot_fig=True, save_fig=False, file_name='nn_diagram.png', **kwargs):
    '''
    Draw a Multi-layer Perceptron classifier or regressor in image.

    Parameters
    ----------
    model: the sklearn.neural_network estimator or a Keras model.
    left : [float] The center of the leftmost node(s) will be placed here. By default, 0.1.
    right : [float] The center of the rightmost node(s) will be placed here. By default, 0.9.
    bottom : [float] The center of bottommost node(s) will be placed here. By default, 0.1.
    top : [float] The center of the topmost node(s) will be placed here. By default, 0.9.
    figsize : [int, int] a method used to change the dimension of plot window, width, height 
                    in inches (default figsize=(15,10)).
    lab_fontsize : [int] Size of text font for annotation of nodes and edges. By default, 12.
    wt_fontsize : [int] Size of text font for annotation of weights. By default, 10.
    edges_color : [list of str] The color of the edges for negative and positive weights.
                    By default, ['gray','k'].
    nodes_color : [str] The color of the nodes. By default, 'w'.
    features_names : [list of str] Names of features seen during fit of model or a custom names 
                    for annotation of Inputs. If None, then the inputs are annotated by the text
                    of: X_1, X_2, ... X_n. By default, None.
    target_names : [list of str] Names of targets seen during fit of model or a custom names 
                    for annotation of Outputs. If None, then the outputs are annotated by the 
                    text of: y_1, y_2, ... y_n. By default, None.
    plot_fig : [bool] If true, displays the figure otherwise the figure is not.
                    By default, plot_fig=True.
    save_fig : [bool] If True, save the current figure in file. By default, save_fig=False.
    file_name : [str] File name for saving the current figure. By default, 'nn_diagram.png'
    **kwargs : other arguments for matplotlib.pyplot.savefig.

    Returns
    ----------
    Plot of the Multi-layer Perceptron classifier or regressor.
    '''
    # The main parameters of the plot
    if 'keras' in str(type(model)):
        layer_sizes=[]
        for i in range(len(model.layers)):
            layer_sizes.append(model.layers[i].get_config().get('units'))
        layer_sizes.insert(0,model.layers[0].get_config().get('batch_input_shape')[1])
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        model_coefs = []
        for layer in model.layers: model_coefs.append(layer.get_weights())
        model_coefs = reduce(lambda x, y: x+y, model_coefs)
        coefs_ = [el for i, el in enumerate(model_coefs) if i%2==0]
        intercepts_ = [el for i, el in enumerate(model_coefs) if i%2]
        n_iter_ = model.optimizer.iterations.numpy()
        loss_ = model.get_metrics_result().get('loss').numpy()
        activation=[]
        for i in range(len(model.layers)):
            activation.append(model.layers[i].get_config().get('activation'))
        activation = list(np.unique(activation))
        solver = model.optimizer.__class__.__name__
        mod_str = str(type(model))
        name_model = 'Multi-layer Perceptron built by {}'.format(mod_str[mod_str.find(' ')+1 : mod_str.find('>')])
    if 'sklearn' in str(type(model)):
        layer_sizes=[model.n_features_in_,*model.get_params()['hidden_layer_sizes'],model.n_outputs_]
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        coefs_ = model.coefs_
        intercepts_ = model.intercepts_
        n_iter_ = model.n_iter_
        loss_ = model.loss_
        activation = model.get_params()['activation']
        solver = model.get_params()['solver']
        name_model = 'Multi-layer Perceptron built by {}'.format(str(model).split("(")[0])
    if features_names is not None:
        name_features_ = features_names   
    fig = plt.figure(figsize=figsize, facecolor='w')
    ax = fig.add_subplot()
    plt.suptitle(name_model, fontsize=lab_fontsize)
    ax.set_ylim(0.1, 1)
    ax.axis('off')
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8.,
                                color=nodes_color, ec='k', zorder=4)
            if n == 0:
                if features_names is not None:
                    plt.text(left-0.18, layer_top - m*v_spacing + 0.005, name_features_[m], fontsize=lab_fontsize)
                if features_names is None:
                    plt.text(left-0.18, layer_top - m*v_spacing + 0.005, r'$X_{'+str(m+1)+'}$', fontsize=lab_fontsize)
                plt.text(n*h_spacing + left, layer_top - m*v_spacing+(v_spacing/8.+0.01*v_spacing), \
                    r'$I_{'+str(m+1)+'}$', fontsize=lab_fontsize)
            elif (n_layers >= 3) & (n < n_layers-1):
                plt.text(n*h_spacing + left, layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing), \
                    r'$H_{'+str(m+1)+'}$', fontsize=lab_fontsize)
            elif n == n_layers-1:
                plt.text(n*h_spacing + left, layer_top - m*v_spacing+(v_spacing/8.+0.01*v_spacing), \
                    r'$O_{'+str(m+1)+'}$', fontsize=lab_fontsize)
                if target_names is not None:
                    plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, target_names[m], fontsize=lab_fontsize)
                if target_names is None:
                    plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, r'$y_{'+str(m+1)+'}$', fontsize=lab_fontsize)
            ax.add_artist(circle)
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., color=nodes_color, ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), y_bias, r'$B_{'+str(n+1)+'}$', fontsize=lab_fontsize)
            ax.add_artist(circle)   
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                if coefs_[n][m, o] < 0:
                    col = edges_color[0]
                if coefs_[n][m, o] >= 0:
                    col = edges_color[1]
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], \
                                  linewidth=abs(coefs_[n][m, o])*2.5, c=col)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                plt.text( xm1, ym1,\
                         str(round(coefs_[n][m, o],4)),\
                         rotation = rot_mo_deg, \
                         fontsize = wt_fontsize)
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], c='k')
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
            plt.text( xo1, yo1,\
                 str(round(intercepts_[n][o],4)),\
                 rotation = rot_bo_deg, \
                 fontsize = wt_fontsize)    
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
    # Record the n_iter_ and loss
    fs=(figsize[0]/figsize[1]*10)
    if fs<lab_fontsize:
        fst=lab_fontsize
    else:
        fst=fs
    plt.text(left + (right-left)/10., bottom - 0.005*v_spacing, \
             'Iterations: '+str(n_iter_)+'    Loss: '+ str(round(loss_, 6))+ \
             '    Activation: {}'.format(activation)+ \
             '    Optimizer: {}'.format(solver), fontsize = fst)
    if save_fig==True:
        fig.savefig(file_name, **kwargs)
    if plot_fig == False:
        plt.close(fig)
    plt.show()


def progress(it,total,buffer=30):
        """
        A progress bar is used to display the progress of a long running Iterations of 
        function, providing a visual cue that processing is underway.
        """
        percent = 100.0*it/(total+1)
        sys.stdout.write('\r')
        sys.stdout.write("Search: [\033[34m{:{}}] {:>3}% ".format('█'*int(percent/(100.0/buffer)),buffer, int(percent)))
        sys.stdout.flush()
        time.sleep(0.001)


def Rait_Test(input_data,method=1):
    """
    Rait's Test (3-sigma criterion) for Outlier Detection
    
    Parameters
    ----------
    data : [Series] vector of data to be verified for outliers.
    method : test's method: 1 - classic test (default), 2 - loyal test.

    Returns
    ----------
    Value is an outlier
    """
    n=len(input_data)
    rmax=max(input_data)
    rmin=min(input_data)
    sdata=input_data.copy()
    min_ind=np.argmin(input_data)
    max_ind=np.argmax(input_data)
    sdata=sdata.drop([max_ind,min_ind])
    mean=np.mean(sdata)
    std=np.std(sdata)
    if method==1:
        if abs(mean-rmax)>3*std:
            print(f"{rmax} - выброс")
        if abs(mean-rmin)>3*std:
            print(f"{rmin} - выброс")
        
    if method==2:
        if n >6 and n <= 100:
            if abs(mean-rmax)>4*std:
                print(f"{rmax} - выброс")
        if n >100 and n <= 1000:
            if abs(mean-rmax)>4.5*std:
                print(f"{rmax} - выброс")
        if n >1000 and n <= 10000:
            if abs(mean-rmax)>5*std:
                print(f"{rmax} - выброс")
        if n >6 and n <= 100:
            if abs(mean-rmin)>4*std:
                print(f"{rmin} - выброс")
        if n >100 and n <= 1000:
            if abs(mean-rmin)>4.5*std:
                print(f"{rmin} - выброс")
        if n >1000 and n <= 10000:
            if abs(mean-rmin)>5*std:
                print(f"{rmin} - выброс")


def regressor_combine(x,Y,func,test_size=0.3,n_splits=5,progress_bar=None,verbose=True,param_grid=None,random_state_tts=None,random_state_KF=None,random_state_f=None):
    """
    Search for the best combination of features in data using ensemble methods.

    Parameters
    ----------
    x : [DataFrame] The training input samples for searching for the best combination
                of features for use in regression methods.
    Y : [array] The target values (class labels) as integers or strings.
    func : a function from the module sklearn.ensemble or sklearn.linear_model.
    test_size : [float or int] If float, should be between 0.0 and 1.0 and represent 
                the proportion of the dataset to include in the test split. If int, 
                represents the absolute number of test samples. By default, 0.3.
    n_splits : [int] Number of folds. Must be at least 2. By defailt, 5.
    progress_bar : [bool] If True to show iterations of algorithm. By default, None.
    verbose : [bool] If True to show number of combinations of features in the data.
                By default, True.
    param_grid : [str] The function's ('func') parameter that changes during the operation of the
                algorithm. By default, None.
    random_state_tts : Controls the shuffling applied to the data before applying the split
                for sklearn.model_selection.train_test_split. By defailt, None.
    random_state_KF : Controls affects the ordering of the indices, which controls the randomness 
                of each fold for sklearn.model_selection.KFold. By defailt, None.
    random_state_f : Controls the randomness of the estimator, which indicated in the func.
                By defailt, None.

    Returns
    ----------
    The table with columns:
    'Parameter' : parameter values of function (if param_grid is not None);
    'R2'        : R2 (coefficient of determination) regression score function;
    'MAE'       : Mean absolute error regression loss;
    'MSE'       : Mean squared error regression loss;
    'ME'        : The max_error metric calculates the maximum residual error;
    'loss'      : loss when testing the model;
    'features'  : combination of features in data.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
    from itertools import combinations
    from sklearn.model_selection import KFold, cross_val_score
    nc=list(x.columns)
    y=[]
    for i in range(2,len(nc)+1):
        y1=list(combinations(nc, i))
        y.extend(y1)
    if verbose is True:
        print('%d combinations of features in the data' % (len(y)))
    if param_grid==None:
        col_names =  ['R2', 'MAE', 'MSE', 'ME', 'loss', 'features']
    else:
        col_names =  ['Parameter ', 'R2', 'MAE', 'MSE', 'ME', 'loss', 'features']
        grid=ParameterGrid(param_grid)
    my_df  = pd.DataFrame(columns = col_names)
    if verbose is True and param_grid != None:
        print('%d combinations will be checked in total' % (len(y)*len(grid)))
    for j in range(0,len(y)):
        X=x[list(y[j])]
        D=pd.concat([X,Y],axis=1)
        train,test = train_test_split(D,test_size=test_size,random_state=random_state_tts)
        X_train, Y_train = train[train.columns[:-1]], train[train.columns[-1]]
        X_test, Y_test = test[test.columns[:-1]], test[test.columns[-1]]
        if param_grid != None:
            for param in grid:
                regr = func(random_state=random_state_f,**param).fit(X_train, Y_train)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_KF)
                results = cross_val_score(regr, X_train, Y_train, cv=kf)
                Y_pred = regr.predict(X_test)
                r2 = round(regr.score(X_test, Y_test), 3)
                mae=round(mean_absolute_error(Y_test, Y_pred), 3)
                mse=round(mean_squared_error(Y_test, Y_pred), 3)
                me=round(max_error(Y_test, Y_pred),3)
                loss=round(regr.loss_, 3)
                nn = list(X.columns)
                col_names =  ['Parameter ', 'R2', 'MAE', 'MSE', 'ME', 'loss', 'features']
                p_data = [(''.join('{}:{} '.format(key, val) for key, val in param.items()))]
                df2 = pd.DataFrame({'Parameter ': p_data, 'R2': r2, 'MAE': mae, 'MSE': mse, 'ME': me, 'loss': loss, 'features': [nn]})
                my_df = pd.concat([my_df, df2],ignore_index=True)
                my_df = my_df.reset_index(drop=True)
                my_df = my_df.rename(index = lambda x: x + 1)
                if progress_bar is True:
                    progress(my_df.shape[0]+1,len(y)*len(grid))
        else:
            regr = func(random_state=random_state_f)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                regr.fit(X_train, Y_train)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_KF)
                results = cross_val_score(regr, X_train, Y_train, cv=kf)
            Y_pred = regr.predict(X_test)
            r2 = round(regr.score(X_test, Y_test), 3)
            mae=round(mean_absolute_error(Y_test, Y_pred), 3)
            mse=round(mean_squared_error(Y_test, Y_pred), 3)
            me=round(max_error(Y_test, Y_pred),3)
            loss=round(regr.loss_, 3)            
            nn = list(X.columns)
            col_names =  ['R2', 'MAE', 'MSE', 'ME', 'loss', 'features']
            df2 = pd.DataFrame({'R2': r2, 'MAE': mae, 'MSE': mse, 'ME': me, 'loss': loss, 'features': [nn]})
            my_df = pd.concat([my_df, df2],ignore_index=True)
            my_df = my_df.reset_index(drop=True)
            my_df = my_df.rename(index = lambda x: x + 1)
            if progress_bar is True:
                progress(my_df.shape[0]+1,len(y))
    return my_df


def scale(data, center=True, scale=True):
    """
    This is a generic function which centers and scales the columns
    of a dataframe or array.
    
    Parameters
    ----------
    data :   [DataFrame, array] dataframe or array for centering, scaling.
    center : [bool] If True (default) then centering is done by subtracting
              the column means of data from their corresponding columns, 
              and if center=False, no centering is done.
    scale :  [bool] If True (default) then scentered columns of the 
              dataframe/array is divided by the root mean square. 
              If scale=False, no scaling is done.
    
    Returns
    ----------
    Dateframe or array which scaled and/or centered and mean values by columns.
      
    """
    x = data.copy()
    xsc=np.mean(x,axis=0)
    if center:
        x -= np.mean(x,axis=0)
    if scale and center:
        x /= np.std(x,axis=0)
    elif scale:
        x /= np.sqrt(np.sum(np.power(x,2),axis=0)/(x.shape[0]-1))
    return x, xsc
