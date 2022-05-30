########Libaray########

import os
import glob
from os.path import join as opj

import numpy as np
import pandas as pd 
from tqdm import tqdm
from easydict import EasyDict
from torch.cuda.amp import autocast
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from dataloader import *
from network import *

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings(action='ignore')

########Setting########

sub = pd.read_csv('files/sample_submission.csv')
bad_df = pd.read_csv('../data/train_df_bad.csv')  

le_bad = LabelEncoder() #le_bad
bad_df['label'] = le_bad.fit_transform(bad_df['label'])

good = le_bad.transform([label for label in le_bad.classes_ if 'good' in label]) #30개
ngood = le_bad.transform([label for label in le_bad.classes_ if not 'good' in label])

train_df = pd.read_csv('../data/train_df.csv')
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
good2 = le.transform([label for label in le.classes_ if 'good' in label]) # 88개

# 앙상블 예측 함수
def get_preds(li, good=good, ngood=ngood, good2=good2, le=le):
    ww = np.array([np.load(i) for i in li]) # 8 x 2514 x 30
    w = ww.mean(axis=0) # 2514 x 30
    w_maxs = np.max(w, axis=1)
    w_preds = np.argmax(w, axis=1)

    df_k2 = pd.DataFrame(data = w_maxs, columns=['max'])
    df_k2['preds'] = w_preds
    df_k2['label'] = le.inverse_transform(w_preds) #string

    bad2 = np.load('files/effb4_bad_5fold.npy') #2514 x 30

    bad2_maxs = np.max(bad2, axis=1)
    bad2_preds = np.argmax(bad2, axis=1)
    df_bad2 = pd.DataFrame(data = bad2_maxs, columns=['max'])
    df_bad2['preds'] = bad2_preds

    # good-bad에서 bad로 예측하거나 good으로 예측해도 softmax값이 0.999999보다 작은 인덱스들 추출
    idx2 = np.array(df_bad2[((df_bad2['preds'].isin(good)) & (df_bad2['max'] <0.999999)) | df_bad2['preds'].isin(ngood)].index)

    #위에서 구한 인덱스들 중에서 예측 레이블이 good인 경우면 2번째 높은 레이블로 변경
    idx_bad2 = np.array(df_k2.loc[idx2][df_k2['label'].isin(le.inverse_transform(good2))].index)
    p_bad2 = np.argsort(w, axis=1)[idx_bad2, -2]

    df_k2['label'].iloc[idx_bad2]= le.inverse_transform(p_bad2)

    return df_k2['label'].values

li = glob.glob('files/softmax_*.npy')
sub['label'] = get_preds(li, good=good, ngood=ngood, good2=good2, le=le)
sub.head()

########Use One-class Classification model########

def Postprocessing_oneclass(cls, sub, npys):
    df_sub = sub.copy()
    idxLst = [df_sub.iloc[idx]['index'] for idx in range(len(df_sub)) if cls in df_sub.iloc[idx]['label']]

    if not npys:
        raise AssertionError('npys must not be empty') 
    # 단일모델 예측 : 기존 모델 예측값 대신 단일 모델 예측값으로 전부 변경
    elif len(npys) == 1:
        path = npys[0]
        p = np.load(path, allow_pickle=True)
        df_sub.loc[idxLst,'label'] = p 

    # 하드보팅 예측 : 단일 모델들의 예측값과 원래의 예측값에 대하여 hard voting
    else:
        df = df_sub[df_sub['index'].isin(idxLst)]

        for path in npys:
            num = os.path.basename(path).split('.')[0][-3:]
            p = np.load(path, allow_pickle=True)
            df[f'pred_{num}'] = p

        for i in range(len(df)):
            label_pred_list = [df.iloc[i,1],df.iloc[i,2],df.iloc[i,3],df.iloc[i,4]]
            if (len(Counter(label_pred_list).most_common(2)) >1) and (Counter(label_pred_list).most_common(2)[1][1] == 2):

                newlabel = df_sub.loc[df.iloc[i]['index'],'label']

            else:
                newlabel = max(label_pred_list, key=label_pred_list.count)

            df_sub.loc[df.iloc[i]['index'],'label'] = newlabel

    return df_sub

sub1 = Postprocessing_oneclass('toothbrush', sub, ['files/toothbrush_220.npy','files/toothbrush_221.npy','files/toothbrush_222.npy'])
sub2 = Postprocessing_oneclass('zipper', sub1, ['files/zipper_254.npy', 'files/zipper_255.npy', 'files/zipper_256.npy'])

sub2.to_csv('./best_ensemble2.csv',index=False)

# sub1 = Postprocessing_oneclass('toothbrush', sub, ['files/toothbrush_220.npy'])
# sub2 = Postprocessing_oneclass('zipper', sub1, ['files/zipper_254.npy', 'files/zipper_255.npy', 'files/zipper_256.npy'])

# sub2.to_csv('./best_ensemble2_zipper_tooth.csv',index=False)
