#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser
import random
import timeit
import sqlite3

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor

import gpumangrove.utils as gmu


def process(args):
    gmu.must_not_exist(args.o)
    print("Processing DBs")
    df = gmu.join_features_measurements(args.fdb,args.mdb, grouping=args.g, aggregator=args.a)
    gmu.process_features(feature_df=df, db_path=args.o)


def filter(args):
    gmu.must_not_exist(args.o)
    print("Filtering samples...")
    if args.t is None:
        threshold = 100
    else:
        threshold = int(args.t)

    gmu.filter_samples(args.i, threshold, pickle_path=args.o)


def cv(args):
    gmu.must_not_exist(args.o)
    gmu.must_not_exist(args.r)

    if 'time' in str.lower(args.r):
        print("Performing cross-validation for time...")
        estimator = TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_jobs=24),
                                           func=np.log,
                                           inverse_func=np.exp)
    elif 'power' in str.lower(args.r):
        print("Performing cross-validation for power...")
        estimator = ExtraTreesRegressor(n_jobs=24)

    scorer = gmu.neg_mape
    # TODO load yml param grid file

    if 'time' in str.lower(args.r):
        param_grid = {
            'regressor__bootstrap': [False],
            'regressor__max_features': [None, 'log2', 'sqrt'],
            'regressor__criterion': ['mse', 'mae'],
            'regressor__n_estimators': [128, 256, 512, 1024]
        }
    elif 'power' in str.lower(args.r):
        param_grid = {
            'bootstrap': [False],
            'max_features': [None, 'log2', 'sqrt'],
            'criterion': ['mse', 'mae'],
            'n_estimators': [128, 256, 512, 1024]
        }


    dataset = pickle.load(open(args.i, "rb"))
    X, y = gmu.get_xy(dataset)
    model, cv_scores = gmu.nested_cv(X, y, estimator, scorer, param_grid,
        num_trials=int(args.t),
        n_splits=int(args.s),
        n_high=int(args.k))

    # for item in cv_scores:
    #    print(pd.DataFrame(item["gs_scores"]))

    if args.o is not None:
        pickle.dump(model, open(args.o, "wb"))
    if args.r is not None:
        pickle.dump(cv_scores, open(args.r,"wb"))



def loo(args):
    gmu.must_not_exist(args.o)
    print("Performing leave one out prediction")

    dataset = pickle.load(open(args.i, "rb"))
    X, y = gmu.get_xy(dataset)
    model = pickle.load((open(args.m, "rb")))
    gmu.loo(X, y, model, args.o)


def ablation(args):
    gmu.must_not_exist(args.o)

    dataset = pickle.load(open(args.i, "rb"))
    model = pickle.load(open(args.m, "rb"))
    gmu.ablation(dataset, model, args.o)


def timemodel(args):
    max_depth = list()
    max_leaf_nodes = list()
    model = pickle.load(open(args.m, "rb"))

    if 'time' in str.lower(args.m):
        model.regressor.n_jobs=1
        model.check_inverse = False
        for tree in model.regressor_.estimators_:
            max_depth.append(tree.tree_.max_depth)
            max_leaf_nodes.append(tree.tree_.node_count)
    elif 'power' in str.lower(args.m):
        model.n_jobs=1
        for tree in model.estimators_:
            max_depth.append(tree.tree_.max_depth)
            max_leaf_nodes.append(tree.tree_.node_count)
    print(model)
    print("Average maximum depth: %0.1f" % (sum(max_depth) / len(max_depth)))
    print("Average count of nodes: %0.1f" % (sum(max_leaf_nodes) / len(max_leaf_nodes)))


    dataset = pickle.load(open(args.i, "rb"))
    X, y = gmu.get_xy(dataset)

    t = np.zeros(args.n)
    for i in range(args.n):
        features = X.sample(random_state=random.randint(0,1e9))
        t[i] = timeit.timeit('model.predict(features)', globals=locals(), number=10) / 10

    print('Min, Max, Average:')
    print(t.min(), t.max(), t.mean())


def paramstats(args):
    gmu.must_not_exist(args.o)

    df_array = pickle.load(open(args.i,"rb"))

    param_df_list = []

    i = 0
    for entry in df_array:
        df = entry['gs_scores']
        param_df_list.append( pd.DataFrame({'i': np.ones(len(df))*i, 'params': df['params'].astype(str), 'score': df['mean_test_score']}))
        i += 1

    param_df = pd.concat(param_df_list)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
        print("Best Parameters for each Iteration")
        print(param_df.sort_values('score',ascending=False).groupby('i',as_index=False).first().sort_values('score',ascending=False)[['params','score']])
        print("Top Parameter Combinations:")
        print(param_df.sort_values('score',ascending=False).groupby('i',as_index=False).first().groupby('params')['score'].mean().sort_values(ascending=False))

    if args.o is not None:
        pickle.dump(param_df, open(args.o, "wb"))


def convert(args):
    gmu.convert(args)

def predict(args):
    samples_db_path = args.i
    model_path = args.m

    conn = sqlite3.Connection(samples_db_path)
    samples = pd.read_sql_query(
        'select * from samples',
        conn, index_col=['bench','app','dataset','name'])

    y_true = samples['time']
    samples = samples.drop(columns=['index','time'])

    print(samples.columns)

    X = samples.values
    model = pickle.load(open(model_path, "rb"))
    y_pred = model.predict(X)
    results = pd.DataFrame()
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['pred/true'] = y_pred/y_true
    print('Results:')
    print(results)

def main():
    # TODO global random state
    # TODO verbose level argument
    p = ArgumentParser(description=
                       """GPU Mangrove:
                       preprocess and filter datesets, train GPU Mangrove model with cross-validation or leave-one-out, prediction with trained model.""")
    subparsers = p.add_subparsers()

    p_convert = subparsers.add_parser('convert', help='Convert raw cuda flux measurements into summarized database')
    p_convert.add_argument('-i', metavar='<input db>', required=True)
    p_convert.add_argument('-o', metavar='<output db>', required=True)
    p_convert.set_defaults(func=convert)

    p_process = subparsers.add_parser('process', help='Join features and measurements, apply feature engineering')
    p_process.add_argument('--fdb', metavar='<feature db>', required=True)
    p_process.add_argument('--mdb', metavar='<measurement db>', required=True)
    p_process.add_argument('-o', metavar='<output db>', required=True)
    p_process.add_argument('-g', choices=['lconf','lseq'], default='lconf')
    p_process.add_argument('-a', choices=['median','mean'], default='median')
    p_process.set_defaults(func=process)

    p_filter = subparsers.add_parser('filter', help='Limit the amunt of samples which are being user per bench,app,dataset,kernel tuple')
    p_filter.add_argument('-o', metavar='<pickle-output>', required=True)
    p_filter.add_argument('-i', metavar='<sample db>', required=True, help='Input for filtering')
    p_filter.add_argument('-t', metavar='<filter threshold>', help='Max. samples per bench,app,dataset,kernel tuple')
    p_filter.set_defaults(func=filter)

    p_cv = subparsers.add_parser('cv', help='train model using cross-validation')
    p_cv.add_argument('-i', metavar='<samples.pkl>', required=True)
    p_cv.add_argument('-o', metavar='<model-output.pkl>')
    p_cv.add_argument('-r', metavar='<result-output.pkl', help='file to store cross-validation scores in')
    p_cv.add_argument('-t', metavar='<num cross validation trials>', default=5)
    p_cv.add_argument('-s', metavar='<num CV splits>', default=3)
    p_cv.add_argument('-k', metavar='<num top "k" samples', default=5,
                      help='Number of top "k" samples to keep in each split')
    p_cv.add_argument('-p', metavar='<param_grid.yml', help='use definition of param grid in yml file')
    p_cv.set_defaults(func=cv)

    p_loo = subparsers.add_parser('loo', help='train model using leave-one-out')
    p_loo.add_argument('-m', metavar='<model.pkl>', required=True)
    p_loo.add_argument('-i', metavar='<samples.pkl>', required=True)
    p_loo.add_argument('-o', metavar='<loo-predictions.pkl>')
    p_loo.set_defaults(func=loo)

    p_ablation = subparsers.add_parser('ablation', help='TODO')
    p_ablation.add_argument('-m', metavar='<model.pkl>', required=True)
    p_ablation.add_argument('-i', metavar='<samples.pkl>', required=True)
    p_ablation.add_argument('-o', metavar='<ablation-scores.pkl>')
    p_ablation.set_defaults(func=ablation)

    p_timemodel = subparsers.add_parser('timemodel', help='measure prediction latency')
    p_timemodel.add_argument('-m', metavar='<model.pkl>', required=True)
    p_timemodel.add_argument('-i', metavar='<samples.pkl>', required=True)
    p_timemodel.add_argument('-n', metavar='<num repeats>', default=100, type=int)
    p_timemodel.set_defaults(func=timemodel)

    p_paramstats = subparsers.add_parser('paramstats', help='TODO')
    p_paramstats.add_argument('-i', metavar='<cv-results.pkl>', required=True)
    p_paramstats.add_argument('-o', metavar='<ablation-scores.pkl>')
    p_paramstats.set_defaults(func=paramstats)

    p_predict = subparsers.add_parser('predict', help='Predict samples from a sample database with a given pre-trained model')
    p_predict.add_argument('-i', metavar='<samples.db>', required=True)
    p_predict.add_argument('-m', metavar='<model.pkl>', required=True)
    p_predict.set_defaults(func=predict)

    args = p.parse_args()
    try:
        args.func(args)
    except AttributeError:
        p.print_help()



if __name__ == '__main__':
    main()

