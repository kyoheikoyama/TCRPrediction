import pandas as pd
import itertools
import numpy as np
import sys, pickle


def get_df(datapath):
    return pd.DataFrame(pickle.load(open(datapath, "rb")))

def get_df_from_path(p_list):
    return pd.concat([get_df(d) for d in p_list]).reset_index(drop=True)


def get_03_data(datapath='./data/03.VDJdb.tsv'):
    df = pd.read_csv(datapath, sep='\t', header=None)
    df['ASeq'] = df[1].str.split(':').apply(lambda x:x[1])
    df['BSeq'] = df[2].str.split(':').apply(lambda x:x[1])
    df = df.drop(columns=[0,1,2])
    df = df.rename(columns={3:'EpiSeq', 4:'Donor', 5:'MHCA', 6:'EpiSpecies'})
    df['ABPairSeq'] = df['ASeq'] +':' + df['BSeq']
    df['interact'] = 1
    return df


def get_all_combination_and_fill_nagatives(df, donor):
    all_combinations = itertools.product(df.ABPairSeq.unique(), df.EpiSeq.unique())
    all_combinations = pd.DataFrame(all_combinations).rename(columns={0:'ABPairSeq', 1:'EpiSeq'})
    df_all = pd.merge(all_combinations, df, on=['ABPairSeq','EpiSeq'], how='left')
    df_all['interact'].fillna(0, inplace=True)
    df_all['ASeq'] = df_all['ABPairSeq'].str.split(':').apply(lambda x: x[0])
    df_all['BSeq'] = df_all['ABPairSeq'].str.split(':').apply(lambda x: x[1])
    df_all['Donor'] = donor
    return df_all


def get_metrics(ytrue, ypred):
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, precision_recall_curve, recall_score
    precision_1 = precision_score(ytrue, ypred, pos_label=1)
    recall = recall_score(ytrue, ypred, pos_label=1)
    precision_0 = precision_score(1 - ytrue, 1 - ypred, pos_label=1)
    acc = accuracy_score(ytrue, ypred)
    bacc = balanced_accuracy_score(ytrue, ypred)
    (tn, fp), (fn, tp) = confusion_matrix(ytrue, ypred)
    rocaucscore = roc_auc_score(ytrue, ypred)
    print(f"{tn}, {fp}, {fn}, {tp} \t | {acc:.04f}, {bacc:.04f}, {precision_1:.04f}, {precision_0:.04f}, {recall:.04f}, {rocaucscore:.04f}")
    return tn, fp, fn, tp, acc, bacc, precision_1, precision_0, recall
    

def create_plot_with_true_pred(true, pred):
    from matplotlib import pyplot as plt
    from inspect import signature
    from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, precision_recall_curve
    true2, pred2 = 1-true, 1-pred
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
    precision, recall, _ = precision_recall_curve(true, pred)

    # In matplotlib < 1.5, ax2.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(ax2.fill_between).parameters
                   else {})

    ax1.step(recall, precision, color='b', alpha=0.2, where='post')
    ax1.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax1.set(xlabel ='Recall', ylabel='Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(true, pred)
    ax1.set_title('Precision-Recall curve on 1: AP={0:0.4f}'.format(average_precision))

    precision, recall, _ = precision_recall_curve(true2, pred2)
    ax2.step(recall, precision, color='b', alpha=0.2, where='post')
    ax2.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax2.set(xlabel ='Recall', ylabel='Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(true2, pred2)
    ax2.set_title('Precision-Recall curve on 0: AP={0:0.4f}'.format(average_precision))


    fpr, tpr, thresholds = roc_curve(true, pred)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label='ROC')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set(xlabel ='False Positive Rate', ylabel='True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    ax3.set_title('ROC curve (AUC = %0.4f)' % roc_auc_score(true, pred))
    ax3.legend(loc="lower right")
    
    
    precision, recall, thresholds = precision_recall_curve(true, pred)
    if len(thresholds) < len(precision): thresholds = list(thresholds)+[1]
    ax4.plot(list(thresholds), precision)
    ax4.plot(list(thresholds), recall)
    ax4.set_title('precision and recall')
    ax4.set(xlabel ='threshold', ylabel='Value of precision or recall')
    ax4.legend(['precision', 'recall'], loc="lower right")
    plt.show()



def pr_auc_score(y_true, y_score):
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, precision_recall_curve, confusion_matrix
    """
    Generates the Area Under the Curve for precision and recall.
    https://github.com/scikit-learn/scikit-learn/issues/5992
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    args = np.argsort(recall)
    precision, recall = precision[args], recall[args]
    return auc(recall, precision)

def get_infer_report(true, pred, thh=0.5):
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, precision_recall_curve, confusion_matrix
    #     true = df_temp['y_true']
    #     pred = scipy.special.softmax(df_temp[['y_pred_0', 'y_pred_1']], axis=1).iloc[:,1]
    print('- acc', accuracy_score(true, pred > thh))
    print('- balanced_acc', balanced_accuracy_score(true, pred > thh))
    precision_1 = precision_score(true, pred >= thh, pos_label=1)
    precision_0 = precision_score(1 - true, pred <= thh, pos_label=1)
    print('- precision on 1: ', precision_1)
    print('- precision on 0: ', precision_0)


    a = confusion_matrix(true, pred > thh)
    tn, fp, fn, tp = a[0][0], a[0][1], a[1][0], a[1][1]
    print(f'- confusion matrix tn, fp, fn, tp: {tn}, {fp}, {fn}, {tp}')

    try:
        print('- roc_auc_score', roc_auc_score(true, pred))
    except:
        print('  None roc_auc_score ')

    print('- pr_auc_score on 1', pr_auc_score(true, pred))
    print('- pr_auc_score on 0', pr_auc_score(1.0 - true, 1.0 - pred))

    if (true.ravel()==1).all():
        pass
    else:
        create_plot_with_true_pred(true, pred)
#         create_plot_with_true_pred(1.0 - true, 1.0 - pred)

def get_file_paths(bucket, dirname, feat_name, extension='.parquet'):
    import boto3, os
    dataroot = f"s3://{bucket}/"
    objs = boto3.resource("s3").Bucket(f"{bucket}").objects.filter(
        Prefix=f"{dirname}/{feat_name}")
    files = [obj.key for obj in objs
             if not obj.key.startswith(f"{dirname}.ipynb_checkpoints") and obj.key.endswith(extension)]
    files = [os.path.join(dataroot, f) for f in files]
    return files
