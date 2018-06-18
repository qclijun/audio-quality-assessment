# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import re
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report

import librosa

DATA_DIR = Path('data')
CLASS_NAMES = [str(i) for i in range(10)]
PERSONS = ['jackson', 'nicolas', 'theo']
WAV_DIRS = [DATA_DIR / name for name in CLASS_NAMES]

SR = 8000
FRAME_LENGTH = int(25 * 0.001 * SR)  # 25ms
HOP_LENGTH = int(10 * 0.001 * SR)  # 10ms

N_MELS = 40
N_MFCC = 13

SEED = 43

H5DF_FILE = Path('data.h5')


@contextmanager
def timer(title):
    start = time.time()
    yield
    print('{} - done in {:.2f} secs'.format(title, time.time() - start))


def get_wav_files():
    wav_files = [p for wav_dir in WAV_DIRS for p in wav_dir.iterdir() if p.suffix == '.wav']
    return wav_files


def get_label_person(filenames):
    pattern = re.compile('(\d)_(\w+)_\d+.wav')
    labels = []
    persons = []
    for filename in filenames:
        match = pattern.match(filename)
        if not match:
            raise ValueError('error wav file name. ' + filename)
        labels.append(int(match.group(1)))
        persons.append(match.group(2))
    return labels, persons


def split_dataset(df, test_person):
    test_mask = df['person'] == test_person
    y_test = df.loc[test_mask, 'label'].values
    y_train = df.loc[~test_mask, 'label'].values

    df = df.drop(['person', 'label', 'file'], axis=1)
    X_test = df[test_mask].values
    X_train = df[~test_mask].values

    return X_train, y_train, X_test, y_test


def __extract_frame_features_single_file(wav_file):
    y, sr = librosa.load(str(wav_file), sr=SR, dtype=np.float32)
    D = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, window='hann')
    mag, phase = librosa.magphase(D, power=1)
    features = []

    mel_s = librosa.feature.melspectrogram(S=mag ** 2, n_mels=40, fmax=SR)
    freq = librosa.fft_frequencies(sr=SR, n_fft=FRAME_LENGTH)

    #  MFCC shape: (N_MFCC,t) where t is number of frames
    mfcc_feat = librosa.feature.mfcc(S=librosa.power_to_db(mel_s), n_mfcc=N_MFCC)
    features.append(mfcc_feat)

    # MFCC delta1, shape: (N_MFCC, t)
    mfcc_delta = librosa.feature.delta(mfcc_feat)
    features.append(mfcc_delta)

    # MFCC delta2
    # mfcc_delta2 = librosa.feature.delta(mfcc_feat, order=2)
    # features.append(mfcc_delta2)

    # zero crossing rate, shape: (1, t)
    zcr = librosa.feature.zero_crossing_rate(y, FRAME_LENGTH, HOP_LENGTH)
    features.append(zcr)

    # RMS energy, shape: (1, t)
    rmse = librosa.feature.rmse(S=mag)
    features.append(rmse)

    # spectral_centroid, shape: (1, t)
    spec_centroid = librosa.feature.spectral_centroid(S=mag, freq=freq)
    features.append(spec_centroid)

    # spectrual_bandwidth, shape: (1, t)
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=mag, freq=freq, p=2)
    features.append(spec_bandwidth)

    # spectral_flatness, shape: (1, t)
    spec_flatness = librosa.feature.spectral_flatness(S=mag)
    features.append(spec_flatness)

    # spectral_rolloff, shape: (1, t)
    spec_rolloff = librosa.feature.spectral_rolloff(S=mag, freq=freq, roll_percent=0.85)
    features.append(spec_rolloff)

    result = np.concatenate(features, axis=0)  # (num_features, t)

    return result


def __extract_global_features_single_file(wav_file):
    frame_features = __extract_frame_features_single_file(wav_file)
    nobs, minmax, mean, variance, skewness, kurtosis = scipy.stats.describe(frame_features, axis=1)
    # global_features = np.concatenate((mean, variance, skewness, kurtosis), axis=0)  # (4*num_features, )
    global_features = np.concatenate((mean, variance, skewness, kurtosis), axis=0)
    return global_features


def extract_global_features(wav_files):
    global_features = [__extract_global_features_single_file(wav_file) for wav_file in wav_files]

    X = np.vstack(global_features)
    return X


def preprocess():
    wav_files = get_wav_files()
    filenames = [f.name for f in wav_files]
    global_features = extract_global_features(wav_files)

    df = pd.DataFrame(global_features, columns=['feature_' + str(i) for i in range(global_features.shape[1])])

    df['file'] = filenames
    labels, persons = get_label_person(filenames)
    df['label'] = labels
    df['person'] = persons

    df.to_hdf(H5DF_FILE, key='df')
    return df


def load_dataset():
    if H5DF_FILE.exists():
        df = pd.read_hdf(H5DF_FILE, key='df')
    else:
        df = preprocess()
    return df


def lgb_model(X_train, X_test, y_train, y_test):
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_eval = lgb.Dataset(data=X_test, label=y_test)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(CLASS_NAMES),
        'metric': ['multi_logloss'],
        'num_leaves': 31,
        'max_depth': 7,
        'learning_rate': 0.1,
        # 'feature_fraction': 0.95,
        # 'bagging_fraction': 0.95,
        # 'bagging_freq': 5,
        'seed': SEED,
        'verbose': 1,
        # 'max_bin': 512
    }

    feature_name = ['feature_' + str(col) for col in range(X_train.shape[-1])]

    nround = 400
    gbm = lgb.train(params, lgb_train,
                    num_boost_round=nround,
                    valid_sets=(lgb_eval,),
                    valid_names=('eval',),
                    feature_name=feature_name,
                    early_stopping_rounds=10
                    )

    gbm.save_model('lgb.model')

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # loss = log_loss(y_test, y_pred)

    # y_pred_classes = y_pred >= 0.5
    # acc = accuracy_score(y_test, y_pred_classes)
    # print('loss={}, acc={}'.format(loss, acc))

    print(y_test[:10])
    print(y_pred[:10])

    # wav_files, classes = load_data()
    #
    # for wav_file, true_class, pred_class in zip(wav_files, y_test, y_pred_classes):
    #     if true_class != pred_class:
    #         print('{}: true={}, pred={}'.format(wav_file, true_class, pred_class))


    return gbm


def xgb_model(X_train, X_test, y_train, y_test):
    feature_names = ['MFCC_'+str(i) for i in range(N_MFCC)]
    feature_names.extend(['MFCC_delta_' + str(i) for i in range(N_MFCC)])
    feature_names.append('zcr')
    feature_names.append('rmse')
    feature_names.append('spec_centroid')
    feature_names.append('spec_bandwidth')
    feature_names.append('spec_flatness')
    feature_names.append('spec_rolloff')

    names = [nm + '_' + stat for stat in ['mean', 'var', 'skew', 'kurt'] for nm in feature_names]

    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=names)
    dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=names)

    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(CLASS_NAMES),
        'max_depth': 6,
        'subsample': 0.95,
        'colsample_bytree': 0.5,
        'eta': 0.01,
        'silent': 1
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    nrounds = 4000
    bst = xgb.train(params, dtrain, nrounds, evallist, early_stopping_rounds=100)
    # bst = xgb.train(params, dtrain, nrounds, evallist)
    bst.save_model('xgb.model')

    # ax = xgb.plot_importance(bst, max_num_features=50)
    # plt.show()

    # bst = xgb.Booster(params, model_file='xgb.model')
    y_pred_prob = bst.predict(dtest, ntree_limit=bst.best_iteration+1)

    y_pred = np.argmax(y_pred_prob, axis=1)
    print(y_pred[:10])
    print(y_test[:10])
    print('classification report:')
    print(classification_report(y_test, y_pred))

    loss = log_loss(y_test, y_pred_prob)
    print('loss:', loss)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    return bst


def main():
    wav_files = get_wav_files()

    with timer("load_dataset"):
        df = load_dataset()

    X_train, y_train, X_test, y_test = split_dataset(df, PERSONS[-1])


    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ', X_test.shape)

    with timer('Training model'):
        # model = lgb_model(X_train, X_test, y_train, y_test)
        model = xgb_model(X_train, X_test, y_train, y_test)
    # X, y = load_dataset()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    # print('X_train.shape: ', X_train.shape)
    # print('X_test.shape: ', X_test.shape)

    # bst, y_pred = xgb_model(X_train, X_test, y_train, y_test)
    # lgb_model(X_train, X_test, y_train, y_test)

    # loss = log_loss(y_test, y_pred)
    #
    # y_pred_class = y_pred >= 0.5
    # acc = accuracy_score(y_test, y_pred_class)
    # print('loss={}, acc={}'.format(loss, acc))
    #
    # for wav_file, true_class, pred_class in zip(wav_files, y_test, y_pred_class):
    #     if true_class != pred_class:
    #         print('{}: true={}, pred={}'.format(wav_file, true_class, pred_class))




if __name__ == '__main__':
    main()
    # wav_files = get_wav_files()
    # # frame_features = __extract_frame_features_single_file(wav_files[13])
    # # print(frame_features.shape)
    # global_features = __extract_global_features_single_file(wav_files[13])
    # print(global_features.shape)
