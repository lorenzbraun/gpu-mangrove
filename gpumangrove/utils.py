import os
import numpy as np
import pandas as pd
import sqlite3
import pickle
from tqdm import tqdm
import re

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, PredefinedSplit, LeaveOneOut

# TODO this function needs refactoring!
def convert(args):
    CUDAFluxMetricDB = args.i
    FeatureTableDB = args.o

    # Get all Column and Row Labels
    with sqlite3.Connection(CUDAFluxMetricDB) as conn:
        cur = conn.cursor()

        cur.execute('select distinct metric from fluxmetrics')
        metrics = cur.fetchall()
        cur.execute('select distinct bench,app,dataset,lseq from fluxmetrics')
        kernels = cur.fetchall()

    # Consistent Mapping/Order of rows and columns
    row = {}
    count = 0
    for kernel in kernels:
        row[kernel] = count
        count += 1
    col = {}
    count = 0
    for metric in metrics:
        col[metric[0]] = count
        count += 1
    with sqlite3.Connection(CUDAFluxMetricDB) as conn:
        data = np.zeros((len(kernels),len(metrics)))

        cur = conn.cursor()
        cur.execute('select * from fluxmetrics')
        print("Processing Items..")
        for item in tqdm(cur.fetchall()):
            data[row[item[0:4]], col[item[5]]] = item[-1]
            
        # Build Dataframe for Labeled Storage
        df_f = pd.DataFrame(data, 
                      index=pd.Index(kernels, names=['bench', 'app', 'dataset', 'lseq']), 
                      columns=[item[0] for item in metrics])
    # Check for Errors
    if False:
        with sqlite3.Connection(CUDAFluxMetricDB) as conn:
            cur = conn.cursor()
            cur.execute("select * from fluxmetrics where bench=? and app=? and dataset=? and lseq=?", kernels[0])
            res = cur.fetchall()
            print(data[0,:])
            print([x[-1] for x in res])
    # Define how instructions are grouped
    inst_groups = {# Integer Inst
       r'(add|sub|mul|mad|mul24|mad24|sad|div|rem|abs|neg|min|max|popc|clz|bfind|fns|brev|bfe|bfi|dp4a|dp2a)[\w.]*?.[us](\d+)$' : 'int',
       # ToDo: extended precission int inst (not found yet)
       # Floating Inst
       r'(testp|copysign|add|sub|mul|fma|mad|div|abs|neg|min|max|rcp)[\w.]*?.f(\d+)$' : 'float',
       r'(sqrt|rsqrt|sin|cos|lg2|ex2)[\w.]*?.f(\d+)$' : 'float.special',
       # Comparing and Selection inst
       r'(set|setp|selp|slct)[\w.]*?.\w(\d+)$' : 'comp_sel',
       # Logic Inst
       r'(and|or|xor|not|cnot|lop3|shf|shl|shr)[\w.]*?.\w?(\d+|pred)$' : 'logic',
       # Data Movement Inst
       r'(ld.global)(.v[42])?.\w(\d+)$' : 'ld.global',
       r'(ld)(.v[42])?.\w(\d+)$' : 'ld.global',
       r'(ld.shared)(.v[42])?.\w(\d+)$' : 'ld.shared',
       r'(ld.volatile.shared)(.v[42])?.\w(\d+)$' : 'ld.shared',
       r'(ld.const)(.v[42])?.\w(\d+)$' : 'ld.const',
       r'(ld.local)(.v[42])?.\w(\d+)$' : 'ld.local',
       r'(ld.param)(.v[42])?.\w(\d+)$' : 'ld.param',
       r'(st.global)(.v[42])?.\w(\d+)$' : 'st.global',
       r'(st)(.v[42])?.\w(\d+)$' : 'st.global',
       r'(st.shared)(.v[42])?.\w(\d+)$' : 'st.shared',
       r'(st.volatile.shared)(.v[42])?.\w(\d+)$' : 'st.shared',
       r'(st.const)(.v[42])?.\w(\d+)$' : 'st.const',
       r'(st.local)(.v[42])?.\w(\d+)$' : 'st.local',
       r'(st.param)(.v[42])?.\w(\d+)$' : 'st.param',
       r'(mov)[\w.]*?.\w?(\d+|pred)$' : 'mov',
       # Data Conversion
       r'(cvta|cvt)[\w.]*?.\w(\d+)$' : 'cvt',
       # Control Flow
       r'(bra|call|ret|exit)[\w.]*?$' : 'control',
       # Atomic Inst
       r'(atom.global)[\w.]*?(\d+)$' : 'atom.global',
       r'(atom.shared)[\w.]*?(\d+)$' : 'atom.shared',
       # Sync
       r'bar.sync' : 'bar.sync'
       # End
      }

    # Helper function to actually group instructions
    def meltInstructions(insts, inst_groups = inst_groups):
        inst_map = { }

        for inst in insts:
            if inst in { 'gX', 'gY', 'gZ', 'bX', 'bY', 'bZ', 'shm', 'time'}:
                inst_map[inst] = [inst]
                continue
            m = None
            for ex in inst_groups:
                m = re.match(ex, inst)
                if( m is not None):
                    num_groups = len(m.groups())
                    #print(inst, m.groups())
                    if (num_groups >= 3):
                        if m.group(num_groups-1) is None:
                            key = inst_groups[ex]+'.'+m.group(num_groups)
                        else:
                            key = inst_groups[ex]+'.'+str(int(m.group(num_groups-1)[2:])*int(m.group(num_groups)))
                    elif (num_groups >= 2):
                        if m.group(num_groups) == 'pred':
                            key = inst_groups[ex]+'.'+'32'
                        else:
                            key = inst_groups[ex]+'.'+m.group(num_groups)
                    else:
                        key = inst_groups[ex]
                    #print(inst, inst_groups[ex], key, m.groups())
                    if key in inst_map:
                        inst_map[key].append(inst)
                    else:
                        inst_map[key] = [inst]
                    break
            if (m == None):
                if 'misc' in inst_map:
                    inst_map['misc'].append(inst)
                else:
                    inst_map['misc'] = [inst]
                print(inst, '\033[1;31mNo Match!\033[0m')
        return inst_map

    # Produce map for summing up columns
    inst_map = meltInstructions(df_f.columns.values)
    # Create grouped dataframe
    dfg = pd.DataFrame(index=df_f.index, columns=[key for key in inst_map])
    for key in tqdm(inst_map):
        dfg[key] = (df_f[inst_map[key]].sum(axis=1))
    # Analytical Metrics

    feature_map = {}

    for col in [ 'gX', 'gY', 'gZ', 'bX', 'bY', 'bZ', 'shm']:
        feature_map[col] = dfg[col].values

    feature_map['CTAs'] = dfg['gX'].values * dfg['gY'].values * dfg['gZ'].values
    feature_map['BlockThreads'] = dfg['bX'].values * dfg['bY'].values * dfg['bZ'].values
    feature_map['TotalThreads'] = feature_map['CTAs'] * feature_map['BlockThreads']

    def compute_data_vol(df, prefix):
        cols = set()
        for col in df.columns.values:
            if col.startswith(prefix):
                cols.add(col)
        res = None
        for col in cols:
            if res is None:
                res = df[col] * int(col.split('.')[-1])/8
            else: 
                res += df[col] * int(col.split('.')[-1])/8
        return res

    feature_map['V_ldGlobal'] = compute_data_vol(dfg, 'ld.global.')
    feature_map['V_stGlobal'] = compute_data_vol(dfg, 'st.global.')
    feature_map['V_ldShm'] = compute_data_vol(dfg, 'ld.shared.')
    feature_map['V_stShm'] = compute_data_vol(dfg, 'st.shared.')
    feature_map['V_ldParam'] = compute_data_vol(dfg, 'ld.param.')
    feature_map['V_stParam'] = compute_data_vol(dfg, 'st.param.')
    feature_map['V_ldLocal'] = compute_data_vol(dfg, 'ld.local.')
    feature_map['V_stLocal'] = compute_data_vol(dfg, 'st.local.')
    feature_map['V_ldConst'] = compute_data_vol(dfg, 'ld.const.')

    cols_inst = ['float.32', 'logic.64', 'comp_sel.32', 'int.32', 'cvt.32',
           'comp_sel.64', 'mov.32', 'logic.32', 'control',
           'int.64', 'cvt.64', 'ld.global.32', 'ld.param.64',
           'st.global.32', 'ld.shared.32', 'mov.64',
           'st.shared.32', 'ld.param.32', 'bar.sync', 'float.64',
           'ld.const.64', 'ld.const.32', 'misc', 'ld.local.32', 'st.local.32',
           'st.global.64', 'atom.shared.32', 'ld.shared.64', 'st.shared.64',
           'comp_sel.16', 'st.global.8', 'ld.global.8', 'mov.16',
           'ld.global.64', 'int.16', 'ld.local.8', 'st.local.8',
           'st.param.64', 'st.param.32', 'atom.global.32', 'logic.16',
           'ld.global.128', 'ld.local.64', 'st.local.64', 'cvt.16',
           'st.global.128', 'ld.shared.8', 'ld.param.8','float.special.32','float.special.64']

    # Hack to avoid error on missing columns
    for col in cols_inst:
        if col not in dfg.columns:
            dfg[col] = 0
    feature_map['total_inst'] = dfg[cols_inst].values.sum(axis=1)

    for col in cols_inst:
        feature_map[col] = dfg[col].values

    features = pd.DataFrame(feature_map)
    # Add Column with Kernel Names
    df_name = None
    with sqlite3.Connection(CUDAFluxMetricDB) as conn:
        # Build Pandas Dataframe from SQL Query
        df_name = pd.read_sql_query(
            'select distinct bench,app,dataset,lseq,name from fluxmetrics', 
            conn, index_col=['bench','app','dataset','lseq'])
    if features.shape[0] == df_name.shape[0]:
        features = features.join(df_name)
    else:
        print("Shape mismatch!")
   # Dump Dataframe into SQL Table
    with sqlite3.connect(FeatureTableDB) as conn:
        features.to_sql('fluxfeatures',conn)
        conn.commit()


def join_features_measurements(feature_db=None, measurement_db=None, grouping='lconf', aggregator='median'):
    groups = {'lseq': ['bench', 'app', 'dataset', 'lseq'],
              'lconf': ['bench','app','dataset','name',
                        'gX','gY','gZ','bX','bY','bZ','shm',
                        'control','int.32','int.64','total_inst']}

    with sqlite3.Connection(feature_db) as conn:
        df_features = pd.read_sql_query("select * from fluxfeatures", conn)

    if ('time' in str.lower(measurement_db)):
        with sqlite3.Connection(measurement_db) as conn:
            df_time = pd.read_sql_query("select * from kerneltime", conn)
        df_grouped = df_time.merge(df_features, on=['bench','app','dataset','lseq'], how='inner').groupby(groups[grouping])
        df_std = df_grouped.std().reset_index()


    elif ('power' in str.lower(measurement_db)):
        with sqlite3.Connection(measurement_db) as conn:
            df_power = pd.read_sql_query("select * from power_filtered", conn)
        df_grouped = df_power.merge(df_features, on=['bench','app','dataset','lseq'], how='inner').groupby(groups[grouping])

    if aggregator == 'median':
        df = df_grouped.median().reset_index()
    if aggregator == 'mean':
        df = df_grouped.mean().reset_index()

    if ('time' in str.lower(measurement_db)):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
            print(df[df_std['time'] > df['time']][['bench','app','dataset','name']])

    # Drop  lseq
    df.drop(columns=['lseq'], inplace=True)

    return df


def process_features(feature_df, db_path=None):
    df = feature_df

    cols = {}

    # 'Index' Columns
    cols['bench'] = df['bench'].values
    cols['app'] = df['app'].values
    cols['dataset'] = df['dataset'].values
    cols['name'] = df['name'].values

    cols['threads_per_CTA'] = df['bX'] * df['bY'] * df['bZ']
    cols['CTAs'] = df['gX'] * df['gY'] * df['gZ']

    cols['total_instructions'] = df['total_inst']
    # Operation groups
    cols['special_operations'] = df['float.special.32'].values + df['float.special.64'].values
    cols['logic_operations'] = df['logic.32'].values + df['logic.16'].values + df['logic.64'].values
    cols['control_operations'] = df['control'].values
    cols['arithmetic_operations'] = (df['float.32'].values + df['float.64'].values
                                    + df['int.32'].values + df['int.16'].values + df['int.64'].values
                                    + df['mov.32'].values + df['mov.64'].values + df['mov.16'].values + df[
                                        'comp_sel.32'].values
                                    + df['comp_sel.64'].values + df['comp_sel.16'].values
                                    + df['cvt.16'].values + df['cvt.32'].values + df['cvt.64'].values)
    cols['sync_operations'] = df['bar.sync'].values

    # Memory data volume (bytes)
    cols['global_memory_volume'] = df['V_stGlobal'].values + df['V_ldGlobal'].values + df['atom.global.32'].values * 4
    # cols['contant_memory_volume'] = df['V_ldConst'].values
    # cols['local_memory_volume'] = df['V_ldLocal'].values + df['V_stLocal'].values
    if 'V_stParam' not in df.columns:
        df['V_stParam'] = 0
    cols['param_memory_volume'] = df['V_ldParam'].values + df['V_stParam'].values
    cols['shared_memory_volume'] = df['V_ldShm'].values + df['V_stShm'].values
    cols['arithmetic_intensity'] = cols['arithmetic_operations'] / (
                cols['global_memory_volume'] + cols['param_memory_volume'])

    if 'time' in df.columns:
        cols['time'] = df['time']
    elif 'aver_power' in df.columns:
        cols['power'] = df['aver_power']


    res = pd.DataFrame(cols)

    if db_path is not None:
        with sqlite3.connect(db_path) as conn:
            res.to_sql('samples', conn)

    return res


def filter_samples(samples, max_samples_per_kernel=100, random_state=42*31415, pickle_path=None, verbose=0):
    if samples is not pd.DataFrame:
        # Read DB
        with sqlite3.connect(samples) as conn:
            dataset = pd.read_sql_query(con=conn, sql="select * from samples",
                                        index_col=['index'])  # , 'bench', 'app', 'dataset'])
    else:
        dataset = samples

    # Remove Invalid Data
    dataset.dropna(inplace=True)

    # For Kernels with many samples reduce to max_samples_per_kernel
    dataset_small = dataset
    if 'time' in str.lower(pickle_path):
        kernelcount = dataset.groupby(['bench', 'app', 'dataset', 'name'])['time'].count()
    elif 'power' in str.lower(pickle_path):
        kernelcount = dataset.groupby(['bench', 'app', 'dataset', 'name'])['power'].count()
 
    for i, count in kernelcount[kernelcount > max_samples_per_kernel].iteritems():
        if verbose > 0:
            print(i, count)
        sample_set = dataset[(dataset['bench'] == i[0])
                             & (dataset['app'] == i[1])
                             & (dataset['dataset'] == i[2])
                             & (dataset['name'] == i[3])]
        dataset_small.drop(sample_set.index, inplace=True)
        dataset_small = dataset_small.append(sample_set.sample(max_samples_per_kernel, random_state=random_state))


    if 'time' in str.lower(pickle_path):
        # Drop Samples with time <= 0.0
        dataset_small.drop(dataset_small[dataset_small['time'].le(0.0)].index, inplace=True)
    elif 'power' in str.lower(pickle_path):
        # Drop Samples with power <= 0.0
        dataset_small.drop(dataset_small[dataset_small['power'].le(0.0)].index, inplace=True)


    if pickle_path is not None:
        pickle.dump(dataset_small,open(pickle_path, "wb"))

    return dataset_small


def get_xy(sample_df):
    assert ((sample_df.columns.values[-1] == 'time') | (sample_df.columns.values[-1] == 'power')),\
        "Last column of DataFrame must be time or power"
    assert ((sample_df.columns.values[:4]) == ['bench', 'app', 'dataset', 'name']).all(),\
        "The first four columns must be index columns (bench,app,dataset,name"

    return sample_df.iloc[:, 4:-1], sample_df.iloc[:, -1]


def nested_cv(X, y, estimator, scorer, param_grid, num_trials=10, n_splits=3, n_high=5, random_state=42*31415):

    groups = group_samples_by_threshold(y, [1e3, 1e5])

    # Data Storage for CV Scores
    cv_scores = []

    # Arrays to store scores
    nested_scores = np.full(num_trials, -np.Inf)
    # Best regression model (return value)
    rg_best = None

    for i in tqdm(range(num_trials)):
        seed = i * random_state

        inner_cv = PredefinedSplit(split_keep_n_high_grouped(y, groups, folds=n_splits, n_high=n_high, random_state=seed))
        outer_cv = PredefinedSplit(split_keep_n_high_grouped(y, groups, folds=n_splits, n_high=n_high, random_state =seed))

        # Non_nested parameter search and scoring
        rg = GridSearchCV(estimator=estimator, param_grid=param_grid,
                          iid=False, cv=inner_cv, scoring=scorer, return_train_score=True)
        rg.fit(X, y)

        # Nested CV with parameter optimization
        nested_score = cross_val_score(rg.best_estimator_, X=X, y=y, cv=outer_cv, scoring=scorer)

        nested_scores[i] = nested_score.mean()
        if nested_scores.max() == nested_scores[i]:
            rg_best = rg.best_estimator_

        cv_scores.append({'gs_scores':pd.DataFrame(rg.cv_results_).sort_values('mean_test_score')[['params', 'mean_test_score']], 'ns_scores':nested_score})

    return rg_best, cv_scores

def simple_cv(X, y, estimator, scorer, num_trials=10, n_splits=3, n_high=5, random_state=42*31415):

    # Data Storage for CV Scores
    cv_scores = []

    for i in tqdm(range(num_trials)):
        seed = i * random_state

        splitter = PredefinedSplit(split_keep_n_high(y, folds=n_splits, n_high=n_high, random_state=seed))
        # Nested CV with parameter optimization
        cv_scores.append(cross_val_score(estimator, X=X, y=y, cv=splitter, scoring=scorer))

    return cv_scores

def ablation(dataset, estimatior, output_path=None, thresholds=np.power(10, [np.arange(2, 10)])[0], verbose=0):
    scores = []

    for threshold in thresholds:
        if 'time' in dataset.columns:
            X, y = get_xy(dataset[dataset['time'] < threshold])
        elif 'power' in dataset.columns:
            X, y = get_xy(dataset[dataset['power'] < threshold])
            if verbose > 0:
                print("Using samples lower than ", threshold)
                print("X shape: ", X.shape)
                print("y shape: ", y.shape)

            scores.append(simple_cv(X, y, estimatior))

    if output_path is not None:
        pickle.dump(scores, open(os.path.join(output_path), "wb"))

    return scores


def group_samples_by_threshold(vec, thresholds):
    thresholds.sort(reverse=True)
    res = np.zeros(vec.shape)
    i = 0
    for t in thresholds:
        res += (vec >= t)
        i += 1
    return res


def split_keep_n_high(targets, folds=3, n_high=5, random_state=42*31415):
    targets = np.array(targets)
    np.random.seed(random_state)

    idx = set(range(len(targets)))
    idxHigh = targets.argsort()[-n_high:]

    split = np.zeros(targets.shape)

    for i in idxHigh:
        split[i] = -1
        idx.remove(i)

    idx = [i for i in idx]
    np.random.shuffle(idx)
    fold = 0
    for i in idx:
        split[i] = fold
        fold += 1
        fold = fold % folds

    return split
# myY = [ 3. ,  0.5,  1. ,  5. , 6. , 7. , 10. , 10.1]
# print(splitHigh(myY, folds=3, n_high=2))


def split_keep_n_high_grouped(targets, groups, folds=3, n_high=5, random_state=42*31415):
    targets = np.array(targets)
    groups = np.array(groups)
    np.random.seed(random_state)

    idx = set(range(len(targets)))
    idxHigh = targets.argsort()[-n_high:]

    split = np.zeros(targets.shape)

    for i in idxHigh:
        split[i] = -1
        idx.remove(i)

    n_groups = int(groups.max()) + 1
    groupMap = {}
    for g in range(0, n_groups):
        groupMap[g] = []

    i = 0
    for g in groups:
        if i in idx:
            groupMap[g].append(i)
        i += 1

    for g in range(0, n_groups):
        idx = [i for i in groupMap[g]]
        np.random.shuffle(idx)
        fold = 0
        for i in idx:
            split[i] = fold
            fold += 1
            fold = fold % folds

    return split


def mape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_pred - y_true)/y_true)


neg_mape = make_scorer(mape, greater_is_better=False)


def must_not_exist(path):
    if path is not None and os.path.exists(path):
        raise Exception('Path "' + path + '" already exists!')
    return


def loo(X, y, model, output_path=None):
    predictions = np.zeros(y.shape)

    splitter = LeaveOneOut()
    for train_index, test_index in tqdm(splitter.split(X, y)):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        predictions[test_index] = model.predict(X.iloc[test_index])

    if output_path is not None:
        pickle.dump(predictions, open(os.path.join(output_path), "wb"))

    return predictions


#myY = [ 3. ,  0.5,  1. ,  5. , 6. , 7. , 10. , 10.1]
#print(splitHighGrouped(myY, [1,0,0,1,1,2,2,2],folds=3, n_high=2))
