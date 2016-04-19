import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from pandas import read_csv, DataFrame, Series
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

TRAIN_FILENAME = '~/kaggle/santander/train.csv'
TEST_FILENAME = '~/kaggle/santander/test.csv'
RAND_SEED = 65535

CRITERION_TO_LOG = 1e2
KCV = 5

def itercols(dataframe):
    for col in dataframe.columns:
        yield (col, dataframe[col])

def read_train_test(fntrain, fntest):
    return read_csv(fntrain), read_csv(fntest)

def scale_train(dataframe):
    #log scaling features with higher than criterion_to_log difference between min and max
    cols_to_log = [kv[0] for kv in filter(lambda nc: np.abs(nc[1].min() - nc[1].max()) > CRITERION_TO_LOG, itercols(dataframe))]
    offsets = {}
    for col in cols_to_log:
        offset = np.abs(dataframe[col]).max() + np.exp(1)
        offsets[col] = offset
        dataframe[col] = np.log(dataframe[col] + offset)

    #standardizing
    means = {}
    deviations = {}
    for col in dataframe.columns:
        ser = dataframe[col]
        
        mean = ser.mean()
        if np.abs(mean) < 1e-10:
            continue
        
        dev = ser.std(ddof=1)
        if np.abs(dev) < 1e-10:
            continue
                
        means[col] = mean
        deviations[col] = dev
        dataframe[col] = (ser - mean) / dev

    return dataframe, cols_to_log, offsets, means, deviations

def scale_test(dataframe, cols_to_log, offsets, means, deviations):
    #log scaling test samle
    for col in cols_to_log:
        offset = offsets[col]
        dataframe[col] = np.log(dataframe[col] + offset)

    #standardizing
    for col in means:
        mean, dev = means[col], deviations[col]
        dataframe[col] = (dataframe[col] - mean) / dev

    return dataframe

def score(X_train, Y_train, X_test, Y_test, params):
    print "params "
    print params
    model_xgb = xgb.XGBClassifier(\
        nthread = -1,\
        n_estimators = 560,\
        max_depth = 5,\
        learning_rate = 0.0202048,\
        min_child_weight = params['min_child_weight'],\
        subsample = params['subsample'],\
        gamma = params['gamma'],\
        colsample_bytree = params['colsample_bytree']\
    )
    
    scores = np.array(cross_validation.cross_val_score(model_xgb, X_train, Y_train, cv = KCV, n_jobs = -1))
    print "cross-val scores"
    print scores
    
    #fitted_xgb = model_xgb.fit(X_train, Y_train)
    #gb_probas = fitted_xgb.predict_proba(X_test)
    #fpr, tpr, _ = roc_curve(Y_test, gb_probas[:, 1])
    #roc_auc = auc(fpr, tpr)
    mean = scores.mean()
    std = scores.std()
    retval = (1.0 - mean) * std

    print "value"
    print retval
    return {'loss': retval, 'mean': mean, 'std': std, 'status': STATUS_OK}

def optimize(X_train, Y_train, X_test, Y_test):
    space = {\
        'min_child_weight' : hp.choice('min_child_weight', np.arange(1, 10, 1, dtype = np.int64)),\
        'subsample' : hp.quniform('subsample', 0.6815, 0.6816, 1e-5),\
        'gamma' : hp.quniform('gamma', 0.5, 1.0, 0.01),\
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.701, 0.702, 1e-4),\
    }
    best = fmin(lambda params: score(X_train, Y_train, X_test, Y_test, params), space, algo = tpe.suggest, trials = Trials(), max_evals = 1000)
    print best


def main():
    data, testdata = read_train_test(TRAIN_FILENAME, TEST_FILENAME)
    cdata, ctestdata = data.drop(['ID', 'TARGET'], axis=1), testdata.drop(['ID'], axis=1)

    Y = data['TARGET']
    X = cdata
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = RAND_SEED)
    optimize(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    main()