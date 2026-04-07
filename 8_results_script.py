'''
This script:
1) imports all dataframes, results & functions
2) Sets up calculations
3) Calculates metrics (R2, RMSE) for both tuned and non-tuned models
4) Finds the knee point for each impact
5) Plots performance vs predictor counts
6) Computes SHAP values
7) Plots predictor contributions
8) Saves SI-related material
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge 
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import os
import copy
from sklearn.metrics import make_scorer, mean_squared_error
from kneed import KneeLocator
import shap
import geopandas as gpd
from matplotlib.lines import Line2D
from IPython.display import display, HTML
np.random.seed(12)

def merge_dicts_double(dict1, dict2):
    '''    
    Recursively merges two dictionaries. Values from dict2 overwrite dict1.
    '''
    merged = copy.deepcopy(dict1)  
    for key, value in dict2.items():
        if key in merged:  
            if isinstance(value, dict) and isinstance(merged[key], dict):  
                merged[key] = merge_dicts_double(merged[key], value)
            else: 
                merged[key] = value
        else:  
            merged[key] = value
    return merged


def merge_all_dicts(dict_list):
    """
    Merges a list of dictionaries using merge_dicts_double.
    """
    if not dict_list:  
        return {}
    merged_dict = dict_list[0]
    for d in dict_list[1:]:
        merged_dict = merge_dicts_double(merged_dict, d)
    return merged_dict


def extract_key_to_df(df, key):
    """
    Extracts a key from a DataFrame of dicts, returns a Series/DataFrame.
    """
    extracted_values = df.map(lambda x: x.get(key) if isinstance(x, dict) else 0)
    return extracted_values


def encode_groups(df, index):
    """Encodes crop groups for cross-validation."""
    groups = df.loc[index][['crop']]
    enc = OrdinalEncoder()
    return enc.fit_transform(groups)


def train_and_evaluate(
        x_train, y_train, impact, estimator_name, 
        subset_size, best_subset, cv, groups_encoded, scoring
        ):
    """
    Trains and evaluates a model for a given impact, estimator, and subset size.
    Returns mean R2 and RMSE.
    """
    n_features = x_train.shape[1]
    feature_mask = [i in best_subset for i in range(n_features)]
    x_train_subset = x_train.iloc[:, feature_mask].copy()
    transformer = PowerTransformer()
    pipeline = Pipeline([
        ('transformer', transformer),
        ('scaler', MinMaxScaler((0, 1))),
        ('regressor', models[estimator_name])
    ])
    cv_results = cross_validate(
        pipeline, x_train_subset, y_train[impact].copy(),
        cv=cv, groups=groups_encoded, n_jobs=-1, scoring=scoring, return_train_score=False
    )
    return cv_results['test_r2'].mean(), cv_results['test_neg_root_mean_squared_error'].mean() , cv_results['test_rmse_original'].mean()


def train_and_evaluate_wcat(
        x_train, y_train, impact, estimator_name, 
        subset_size, best_subset, cv, groups_encoded, scoring
        ):
    """
    Trains and evaluates a model for a given impact, estimator, and subset size, using categorical
    information as well as numerical.Returns mean R2 and RMSE.
    """
    n_features = x_train.shape[1] - x_train.columns.str.contains('KG_').sum() + 1
    feature_mask = [i in best_subset for i in range(n_features)]
    if feature_mask[-1] == True:
        feature_mask = feature_mask + [True, True, True]
    else:
        feature_mask = feature_mask + [False, False, False]
    x_train_subset = x_train.iloc[:, feature_mask].copy()
    transformer = PowerTransformer()
    pipeline = Pipeline([
        ('transformer', transformer),
        ('scaler', MinMaxScaler((0, 1))),
        ('regressor', models[estimator_name])
    ])
    cv_results = cross_validate(
        pipeline, x_train_subset, y_train[impact].copy(),
        cv=cv, groups=groups_encoded, n_jobs=-1, scoring=scoring, return_train_score=False
    )
    return cv_results['test_r2'].mean(), cv_results['test_neg_root_mean_squared_error'].mean() , cv_results['test_rmse_original'].mean()


def train_and_evaluate_cat_only(
        x_train, y_train, impact, estimator_name, 
        subset_size, best_subset, cv, groups_encoded, scoring
        ):
    """
    Trains and evaluates a model for a given impact, estimator, and subset size, using only
    categorical information. Returns mean R2 and RMSE.
    """
    x_train_subset = x_train.loc[:, (x_train.columns.str.contains('prennial|KG_'))].copy()
    transformer = PowerTransformer()
    pipeline = Pipeline([
        ('transformer', transformer),
        ('scaler', MinMaxScaler((0, 1))),
        ('regressor', models[estimator_name])
    ])
    cv_results = cross_validate(
        pipeline, x_train_subset, y_train[impact].copy(),
        cv=cv, groups=groups_encoded, n_jobs=-1, scoring=scoring, return_train_score=False
    )
    return cv_results['test_r2'].mean(), cv_results['test_neg_root_mean_squared_error'].mean() , cv_results['test_rmse_original'].mean()

def train_and_evaluate_cat_yield(
        x_train, y_train, impact, estimator_name, 
        subset_size, best_subset, cv, groups_encoded, scoring
        ):
    """
    Trains and evaluates a model for a given impact, estimator, and subset size, using categorical and 
    top information (yield for biodiversity and climate footprint, and irrigation for water footprint) only.
    Returns mean R2 and RMSE.
    """
    if ('water' in impact) or ('wu' in impact):
        x_train_subset = x_train.loc[:, (x_train.columns.str.contains('prennial|KG_|irrigation'))].copy()
    else:
        x_train_subset = x_train.loc[:, (x_train.columns.str.contains('prennial|KG_|yield'))].copy()

    transformer = PowerTransformer()
    pipeline = Pipeline([
        ('transformer', transformer),
        ('scaler', MinMaxScaler((0, 1))),
        ('regressor', models[estimator_name])
    ])
    cv_results = cross_validate(
        pipeline, x_train_subset, y_train[impact].copy(),
        cv=cv, groups=groups_encoded, n_jobs=-1, scoring=scoring, return_train_score=False
    )
    return cv_results['test_r2'].mean(), cv_results['test_neg_root_mean_squared_error'].mean() , cv_results['test_rmse_original'].mean()


def compute_global_shap_values(
        x_train, y_train, best_models_knee, 
        best_predictors_all, tuned_estimators_all, transformer
        ):
    """
    Compute global SHAP values for each impact and estimator in best_models_knee.
    Returns a nested dict: global_shap_values[impact][estimator_name] = {feature: mean_abs_shap}
    """

    model_classes = {
        "RF": RandomForestRegressor(n_jobs=-1, random_state=12),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "GBM": HistGradientBoostingRegressor(random_state=12),
        "GLM": Ridge(),
        "ANN": MLPRegressor(random_state=12, max_iter=10000)
    }

    global_shap_values = {}

    for impact, estimators in best_models_knee.items():
        global_shap_values[impact] = {}
        y_train_impact = y_train[impact].copy()
        for estimator_name, best_subset_size in estimators.items():
            arg = tuned_estimators_all[impact][estimator_name][best_subset_size].copy()
            best_subset = best_predictors_all[impact][estimator_name][best_subset_size]
            feature_mask = [i in best_subset for i in range(x_train.shape[1])]
            x_train_subset = x_train.iloc[:, feature_mask].copy()

            # Handle ANN hidden layer sizes
            if estimator_name == "ANN":
                layers = tuple(arg.pop(k) for k in ('i', 'j', 'k') if k in arg)
                arg['hidden_layer_sizes'] = layers if layers else (arg.pop('i'),)

            estimator_obj = model_classes[estimator_name].set_params(**arg)
            pipeline = Pipeline([
                ('transformer', transformer),
                ('scaler', MinMaxScaler((0, 1))),
                ('regressor', estimator_obj)
            ])
            pipeline.fit(x_train_subset, y_train_impact)

            explainer = shap.Explainer(pipeline.predict, x_train_subset)
            shap_values = explainer(x_train_subset)
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            feature_names = x_train_subset.columns.tolist()
            shap_summary = dict(zip(feature_names, mean_abs_shap))
            global_shap_values[impact][estimator_name] = shap_summary

    return global_shap_values

def rmse_original(y_true, y_pred):
    # Inverse transform log values
    y_true_orig = np.exp(y_true)
    y_pred_orig = np.exp(y_pred)
    return np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n","<br>")))

rmse_original_scorer = make_scorer(rmse_original, greater_is_better=False)
perspectives = [ 'P']
found_perspectives = []
impact_order = [
    'Biodiversity Loss', 
    'Water Use', 
    'Climate Change'
    ]

for p in perspectives:
    required_files = [
        f'df_categorical_{p}.pkl',
        f'df_impacts_{p}.pkl',
        f'x_{p}_wcat.pkl',
        f'dict_predictor_importances_{p}_IC_A_wcat.pkl',
        f'dict_tuned_estimators_{p}_IC_A_wcat.pkl'
    ]
    if all(os.path.exists(f) for f in required_files):
        found_perspectives.append(p)

print(f"Found perspectives: {', '.join(found_perspectives)}")

for p in found_perspectives:
    print(f"\nProcessing perspective {p}...")

    # 1) Import necessary dataframes built in previous scripts
    df_categorical = pd.read_pickle(f'df_categorical_{p}.pkl')
    df_impacts = pd.read_pickle(f'y_{p}_wcat.pkl')
    df_inputs = pd.read_pickle(f'x_{p}_wcat.pkl')

    print('df_inputs: ', df_inputs.shape, '\n',
        'df_impacts: ', df_impacts.shape, '\n', 
        'df_categorical: ', df_categorical.shape)

    # Import results and create dictionaries
    prefix_imp = (f"dict_predictor_importances_{p}_bl",
                  f"dict_predictor_importances_{p}_cc",
                  f"dict_predictor_importances_{p}_wu")
    files_imp = [f for f in os.listdir('.') if (f.startswith((prefix_imp))) and ('wcat' in f)]
    dicts_imp = [pickle.load(open(file, 'rb')) for file in files_imp]
    best_predictors_all = merge_all_dicts(dicts_imp)

    prefix_tuned = (f"dict_tuned_estimators_{p}_bl",
                     f"dict_tuned_estimators_{p}_cc",
                       f"dict_tuned_estimators_{p}_wu") 
    files_tuned = [f for f in os.listdir('.') if (f.startswith(prefix_tuned)) and ('wcat' in f)]
    dicts_tuned = [pickle.load(open(file, 'rb')) for file in files_tuned]
    tuned_estimators_all = merge_all_dicts(dicts_tuned)

    # 2) Setup calculations
    groups = df_categorical[['crop']]
    groups_encoded = encode_groups(df_categorical, df_inputs.index)
    y_train = df_impacts
    x_train = df_inputs

    models = {'RF' : RandomForestRegressor(n_jobs = 1, random_state=12),
              'KNN' : KNeighborsRegressor(n_jobs=1),
              'ANN' : MLPRegressor(random_state=12, max_iter=10000),
              'GBM' : HistGradientBoostingRegressor(random_state = 12), 
              'GLM' : Ridge()
              }
                
    best_estimators = {key: {model:{} for model in models.keys()} for key in df_impacts.columns}
    estimators = models.keys()
    list_subset_sizes = list(np.arange(1,17))
    transformer = PowerTransformer()
    
    # 3) Find performance metrics for non-tuned and tuned models
    results_notuning = {
        key_1:{
            key_2:{
                key_3:{
                    'r2': [] , 
                    'rmse':[], 
                    'rmse_original': []
                    } 
                    for key_3 in list_subset_sizes
                    } 
                    for key_2 in estimators
                    } 
                    for key_1 in df_impacts.columns
                    }
    scoring = {'r2': 'r2', 
               'neg_root_mean_squared_error': 'neg_root_mean_squared_error', 
               'rmse_original': rmse_original_scorer
               }

    groups = df_categorical.loc[x_train.index][['crop']]
    groups_encoded = encode_groups(df_categorical, df_inputs.index)
    cv = GroupKFold(n_splits = 5, shuffle = True, random_state = 12)

    for impact in df_impacts.columns:
        for estimator_name in tqdm(estimators):
            if estimator_name == "RF":
                classifier_obj = RandomForestRegressor(n_jobs=-1, random_state = 12)
            elif estimator_name == "KNN":
                classifier_obj = KNeighborsRegressor(n_neighbors= 5, n_jobs=-1)
            elif estimator_name == "GBM":
                classifier_obj = HistGradientBoostingRegressor(random_state=12)
            elif estimator_name == "GLM":
                classifier_obj = Ridge()
            elif estimator_name == "ANN":
                classifier_obj = MLPRegressor(random_state=12, max_iter = 20000)
            for subset_size in list_subset_sizes:

                best_subset  = best_predictors_all[impact][estimator_name][subset_size]            
                (results_notuning[impact][estimator_name][subset_size]['r2'],
                  results_notuning[impact][estimator_name][subset_size]['rmse'],
                    results_notuning[impact][estimator_name][subset_size]['rmse_original']) = train_and_evaluate_wcat(x_train, y_train, impact, 
                                                                                                                 estimator_name, subset_size, 
                                                                                                                 best_subset, cv, groups_encoded, scoring
                                                                                                                 )
    
    groups = df_categorical.loc[x_train.index][['crop']]
    groups_encoded = encode_groups(df_categorical, df_inputs.index)
    cv = GroupKFold(n_splits = 5, shuffle = True, random_state = 12)

    results_tuned = {
        key_1:{
            key_2:{
                key_3:{
                    'r2':[],
                    'rmse':[], 
                    'rmse_original' : []
                    } 
                    for key_3 in list_subset_sizes
                    } 
                    for key_2 in estimators
                    } 
                    for key_1 in df_impacts.columns
                    }
    
    models = {'RF' : RandomForestRegressor(n_jobs = -1, random_state=12),
              'KNN' : KNeighborsRegressor(n_jobs=-1),
              'ANN' : MLPRegressor(random_state=12, max_iter=20000),
              'GBM' : HistGradientBoostingRegressor(random_state = 12), 
              'GLM' : Ridge() 
              }
    estimators = models.keys()

    for impact in df_impacts.columns:
        for estimator_name in estimators:
            for subset_size in list_subset_sizes:
                arg = tuned_estimators_all[impact][estimator_name][subset_size].copy()

                if estimator_name == "ANN":
                    num_layers = arg.pop('num_layers')
                    if num_layers == 1:
                        arg['hidden_layer_sizes'] = (arg.pop('i'),)
                    elif num_layers == 2:
                        arg['hidden_layer_sizes'] = (arg.pop('i'),
                                                      arg.pop('j')
                                                      )
                    else:
                        arg['hidden_layer_sizes'] = (arg.pop('i'),
                                                      arg.pop('j'),
                                                        arg.pop('k')
                                                        )
                n_features = x_train.shape[1] - x_train.columns.str.contains('KG_').sum() + 1
                best_subset  = best_predictors_all[impact][estimator_name][subset_size]
                feature_mask = [i in best_subset for i in range(n_features)]
                feature_mask = feature_mask + [False, False, False]
                x_train_subset = x_train.iloc[:, feature_mask].copy()

                estimator_obj = models[estimator_name]
                estimator_obj = estimator_obj.set_params(**arg)
                pipeline = Pipeline([('transformer', transformer),
                                      ('scaler', MinMaxScaler((0,1))),
                                        ('regressor', estimator_obj) ])
                
                cv_results = cross_validate(
                    pipeline, x_train_subset, y_train[impact].copy(), cv=cv, 
                    groups = groups_encoded, scoring=scoring, n_jobs=-1,  
                    error_score='raise', return_train_score=False, 
                    return_estimator = True
                    )

                results_tuned[impact][estimator_name][subset_size]['r2'] = cv_results['test_r2'].mean()
                results_tuned[impact][estimator_name][subset_size]['rmse'] = cv_results['test_neg_root_mean_squared_error'].mean()
                results_tuned[impact][estimator_name][subset_size]['rmse_dist'] = cv_results['test_neg_root_mean_squared_error']
                results_tuned[impact][estimator_name][subset_size]['r2_dist'] = cv_results['test_r2']
                results_tuned[impact][estimator_name][subset_size]['rmse_original'] = cv_results['test_rmse_original'].mean()

            print('results added for ', impact, ', ', estimator_name)

    df_wu = pd.DataFrame.from_dict(results_tuned['Water Use'])
    df_bl = pd.DataFrame.from_dict(results_tuned['Biodiversity Loss'])
    df_cc = pd.DataFrame.from_dict(results_tuned['Climate Change'])
    df_wu_notuning = pd.DataFrame.from_dict(results_notuning['Water Use'])
    df_bl_notuning = pd.DataFrame.from_dict(results_notuning['Biodiversity Loss'])
    df_cc_notuning = pd.DataFrame.from_dict(results_notuning['Climate Change'])

    df_wu.to_excel(f'df_wu_{p}_wcat.xlsx')
    df_bl.to_excel(f'df_bl_{p}_wcat.xlsx')
    df_cc.to_excel(f'df_cc_{p}_wcat.xlsx')

    # Saving the hyperparams of optimized models for SI
    w = pd.ExcelWriter(f'S6 - optimized_hyperparams_{p}.xlsx')
    impact_order = ['Biodiversity Loss', 'Water Use', 'Climate Change']

    for impact in impact_order:
        df = pd.DataFrame.from_dict(tuned_estimators_all[impact])
        df.to_excel(w, sheet_name= impact)
    w.close()

    # 3.5) Performance of models using only the categorical variables
    results_cat = {
        key_1:{
            key_2:{
                key_3:{
                    'r2': [] , 
                    'rmse':[], 
                    'rmse_original': []
                    } 
                    for key_3 in list_subset_sizes
                    } 
                    for key_2 in estimators
                    } 
                    for key_1 in df_impacts.columns
                    }
    scoring = {'r2': 'r2', 
               'neg_root_mean_squared_error': 'neg_root_mean_squared_error', 
               'rmse_original': rmse_original_scorer
               }

    groups = df_categorical.loc[x_train.index][['crop']]
    groups_encoded = encode_groups(df_categorical, df_inputs.index)
    cv = GroupKFold(n_splits = 5, shuffle = True, random_state = 12)

    for impact in df_impacts.columns:
        for estimator_name in tqdm(estimators):
            if estimator_name == "RF":
                classifier_obj = RandomForestRegressor(n_jobs=-1, random_state = 12)
            elif estimator_name == "KNN":
                classifier_obj = KNeighborsRegressor(n_neighbors= 5, n_jobs=-1)
            elif estimator_name == "GBM":
                classifier_obj = HistGradientBoostingRegressor(random_state=12)
            elif estimator_name == "GLM":
                classifier_obj = Ridge()
            elif estimator_name == "ANN":
                classifier_obj = MLPRegressor(random_state=12, max_iter = 20000)
            for subset_size in list_subset_sizes:

                best_subset  = best_predictors_all[impact][estimator_name][subset_size]            
                (results_cat[impact][estimator_name][subset_size]['r2'],
                  results_cat[impact][estimator_name][subset_size]['rmse'],
                    results_cat[impact][estimator_name][subset_size]['rmse_original']) = train_and_evaluate_cat_only(x_train, y_train, impact, 
                                                                                                                 estimator_name, subset_size, 
                                                                                                                 best_subset, cv, groups_encoded, scoring
                                                                                                                 )
    df_wu_cat = pd.DataFrame.from_dict(results_cat['Water Use'])
    df_bl_cat = pd.DataFrame.from_dict(results_cat['Biodiversity Loss'])
    df_cc_cat = pd.DataFrame.from_dict(results_cat['Climate Change'])

    df_wu_cat.to_excel(f'df_wu_{p}_cat_only.xlsx')
    df_bl_cat.to_excel(f'df_bl_{p}_cat_only.xlsx')
    df_cc_cat.to_excel(f'df_cc_{p}_cat_only.xlsx')

    results_cat = {
        key_1:{
            key_2:{
                key_3:{
                    'r2': [] , 
                    'rmse':[], 
                    'rmse_original': []
                    } 
                    for key_3 in list_subset_sizes
                    } 
                    for key_2 in estimators
                    } 
                    for key_1 in df_impacts.columns
                    }
    scoring = {'r2': 'r2', 
               'neg_root_mean_squared_error': 'neg_root_mean_squared_error', 
               'rmse_original': rmse_original_scorer
               }

    groups = df_categorical.loc[x_train.index][['crop']]
    groups_encoded = encode_groups(df_categorical, df_inputs.index)
    cv = GroupKFold(n_splits = 5, shuffle = True, random_state = 12)

    for impact in df_impacts.columns:
        for estimator_name in tqdm(estimators):
            if estimator_name == "RF":
                classifier_obj = RandomForestRegressor(n_jobs=-1, random_state = 12)
            elif estimator_name == "KNN":
                classifier_obj = KNeighborsRegressor(n_neighbors= 5, n_jobs=-1)
            elif estimator_name == "GBM":
                classifier_obj = HistGradientBoostingRegressor(random_state=12)
            elif estimator_name == "GLM":
                classifier_obj = Ridge()
            elif estimator_name == "ANN":
                classifier_obj = MLPRegressor(random_state=12, max_iter = 20000)
            for subset_size in list_subset_sizes:

                best_subset  = best_predictors_all[impact][estimator_name][subset_size]            
                (results_cat[impact][estimator_name][subset_size]['r2'],
                  results_cat[impact][estimator_name][subset_size]['rmse'],
                    results_cat[impact][estimator_name][subset_size]['rmse_original']) = train_and_evaluate_cat_yield(x_train, y_train, impact, 
                                                                                                                 estimator_name, subset_size, 
                                                                                                                 best_subset, cv, groups_encoded, scoring
                                                                                                                 )
    df_wu_cat = pd.DataFrame.from_dict(results_cat['Water Use'])
    df_bl_cat = pd.DataFrame.from_dict(results_cat['Biodiversity Loss'])
    df_cc_cat = pd.DataFrame.from_dict(results_cat['Climate Change'])

    df_wu_cat.to_excel(f'df_wu_{p}_cat_yield.xlsx')
    df_bl_cat.to_excel(f'df_bl_{p}_cat_yield.xlsx')
    df_cc_cat.to_excel(f'df_cc_{p}_cat_yield.xlsx')

    # 4) We find the knee point for each footprint using the kneedle package
    list_df_impacts = [df_bl, df_wu, df_cc]
    impact_names = ['Biodiversity Loss', 'Water Use', 'Climate Change']
    best_models_knee = {}

    for df_impact, impact_name in zip(list_df_impacts, impact_names):
        y = df_impact.map(lambda x: x['rmse']).max(axis=1)
        x = df_impact.index
        kneedle = KneeLocator(x, y, S=0, curve="concave", direction="increasing", online=False)
        knee_idx = kneedle.knee
        rmse_df = df_impact.map(lambda x: x['rmse'])
        max_model_per_row = rmse_df.idxmax(axis=1)
        best_model_at_knee = max_model_per_row[knee_idx]
        print(f"{best_model_at_knee} gives the maximum RMSE at {knee_idx} predictors for {impact_name}")
        best_models_knee[impact_name] = {best_model_at_knee: knee_idx}
    print(best_models_knee)

    # 5) Plotting performance vs predictor count
    sns.set(font_scale=1.55)
    sns.set_style('whitegrid')
    set2_palette  = sns.color_palette('colorblind')
    custom_colors = {
        'GBM': set2_palette[6],  # Pink
        'RF': set2_palette[3],   # Darker green
        'ANN': set2_palette[9],  # Purple
        'KNN': set2_palette[1],  # Orange
        'LM': set2_palette[2],   # Light green
        }

    impact_order = ['Biodiversity Loss', 'Water Use', 'Climate Change']
    model_order = ['KNN', 'GBM', 'LM', 'RF', 'ANN']

    common_dash_pattern = (2, 2)
    list_mean_dfs = []
    rmses = []
    mean_r2_dfs = []
    mean_rmse_dfs = []

    for impact in impact_order:
        df_notuning = pd.DataFrame.from_dict(
            results_notuning[impact]
            ).rename(columns={'GLM': 'LM'})
        mean_r2_df_notuning = extract_key_to_df(df_notuning, 'r2').fillna(0)
        mean_r2_df_notuning = mean_r2_df_notuning[model_order]

        mean_rmse_df_notuning = extract_key_to_df(df_notuning, 'rmse')
        mean_rmse_df_notuning = np.abs(mean_rmse_df_notuning.fillna(0))
        mean_rmse_df_notuning = mean_rmse_df_notuning[model_order]

        df_tuned = pd.DataFrame.from_dict(
            results_tuned[impact]
            ).rename(columns={'GLM': 'LM'})
        mean_r2_df_tuned = extract_key_to_df(df_tuned, 'r2').fillna(0)
        mean_r2_df_tuned = mean_r2_df_tuned[model_order]

        mean_rmse_df_tuned = extract_key_to_df(df_tuned, 'rmse')[model_order]
        mean_rmse_df_tuned = np.abs(mean_rmse_df_tuned.dropna(axis=1))

        mean_r2_dfs.append(mean_r2_df_tuned)
        mean_rmse_dfs.append(mean_rmse_df_tuned)

    fig_r2, axes_r2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True, squeeze=True)
    for i, (ax, impact, mean_r2_df) in enumerate(zip(axes_r2, impact_order, mean_r2_dfs)):
        sns.lineplot(
            data=mean_r2_df,
            palette=custom_colors,
            linestyle='dashed',
            markers='o',
            dashes=False,
            linewidth=1.5,
            alpha=0.8,
            ax=ax
        )
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Number of predictors')
        if i == 0:
            ax.set_ylabel('R2')
        else:
            ax.set_ylabel('')
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16], minor=False)
        ax.set_title(impact)
        ax.get_legend().remove()

        # Add vertical line and model name for best_models_knee
        if impact in best_models_knee:
            model_name, n_predictors = list(best_models_knee[impact].items())[0]
            ax.axvline(n_predictors, color='k', linestyle='--', linewidth=2, alpha=0.7)
            ylim = ax.get_ylim()
            ax.text(
                n_predictors - 0.2, ylim[1] * 0.95,
                model_name,
                color='k',
                fontsize=18,
                va='top',
                ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )

    fig_r2.tight_layout()
    fig_r2.suptitle('(a)', y=1.02, x=0.53)
    fig_r2.savefig(f'R2_across_impacts_{p}_original_rmse_wcat.png', bbox_inches='tight')

    fig_rmse, axes_rmse = plt.subplots(1, 3, figsize=(15, 5), sharey=True, squeeze=True)
    for i, (ax, impact, mean_rmse_df) in enumerate(zip(axes_rmse, impact_order, mean_rmse_dfs)):
        sns.lineplot(
            data=mean_rmse_df,
            palette=custom_colors,
            linestyle='dashed',
            marker='o',
            dashes=False,
            linewidth=1.5,
            alpha=0.8,
            ax=ax
        )
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('Number of predictors')
        if i == 0:
            ax.set_ylabel('RMSE')
        else:
            ax.set_ylabel('')
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16], minor=False)
        ax.set_title(impact)
        ax.get_legend().remove()

        # Add vertical line and model name for best_models_knee
        if impact in best_models_knee:
            model_name, n_predictors = list(best_models_knee[impact].items())[0]
            ax.axvline(n_predictors, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            ylim = ax.get_ylim()
            ax.text(
                n_predictors -0.2, ylim[1] * 0.95,
                model_name,
                color='k',
                fontsize=18,
                va='top',
                ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )

    fig_rmse.tight_layout()
    fig_rmse.suptitle('(b)', y=1.02, x=0.53)
    fig_rmse.savefig(f'RMSE_across_impacts_{p}_original_rmse_wcat.png',
                      bbox_inches='tight')
    plt.show()

    # Add a separate figure for legend
    legend_fig, legend_ax = plt.subplots(figsize=(15, 1))
    legend_ax.axis('off')
    handles = []
    labels = []
    for model, color in custom_colors.items():
        handles.append(Line2D([0], [0], color=color, marker='o', 
                              linestyle='dashed', linewidth=1.5, markersize=8))
        labels.append(model)
    legend_ax.legend(handles, labels, loc='center', ncol=len(labels), 
                     fontsize=20, frameon=True)
    legend_fig.tight_layout()
    legend_fig.savefig(f'legend_across_impacts_{p}_original_rmse_wcat.png', 
                       bbox_inches='tight')
    plt.show()

    # 6) Compute SHAP values & save them
    global_shap_values = compute_global_shap_values(
        x_train, y_train, best_models_knee, best_predictors_all, 
        tuned_estimators_all, transformer
    )
    with open(f'global_shap_values_{p}_wcat.pkl', 'wb') as f:
        pickle.dump(global_shap_values, f)

    sns.set_style('whitegrid')
    sns.set(font_scale=1.50)
    with open(f'global_shap_values_H_wcat.pkl', 'rb') as f:
        global_shap_values = pickle.load(f)

    global_shap_values['biodiversity footprint'] = global_shap_values.pop('Biodiversity Loss')
    global_shap_values['water footprint'] = global_shap_values.pop('Water Use')
    global_shap_values['GHG footprint'] = global_shap_values.pop('Climate Change')

    footprints = ['biodiversity footprint', 'water footprint', 'GHG footprint']
    w = pd.ExcelWriter(f'S7 - predictor_contributions_{p}_wcat.xlsx')
    for footprint in footprints:
        df = pd.DataFrame(global_shap_values[footprint].values()).T
        df.columns = ['Contribution to prediction of '+ footprint]
        df.to_excel(w, sheet_name= footprint)
    w.close()

    # 7) Plot contribution of various predictors to predictions
    base_colors = {
        'fertilizers': 'tomato',
        'energy':     'darkorange',
        'water':      'steelblue',
        'yield':      'mediumaquamarine',
        'protection': 'violet',
        'location' : 'teal',
        'crop_type': 'dimgrey'
    }

    predictors = {
        'fertilizers': [
            'N_fertiliser_kg', 'P_fertiliser_kg', 
            'K_fertiliser_kg', 'manure_kg', 'micronutrients_kg', 
            'stimulant_kg', 'soil_improvement_kg'
            ],
        'energy': ['fuel_MJ', 'heat_MJ', 'electricity_kWh', 'blahhhhh'],
        'yield': ['yield_kg/ha', 'blah'],
        'water': ['irrigation_m3', 'tap_water_kg'],
        'protection': ['protection_kg', 'blhahh'],
        'location' : ['KG_A', 'KG_B', 'KG_C', 'KG_D'],
        'crop_type' : ['prennial/annual']
    }

    label_mapping = {
        'N_fertiliser_kg': 'Nitrogen fertilizer',
        'P_fertiliser_kg': 'Phosphorus fertilizer',
        'K_fertiliser_kg': 'Potassium fertilizer',
        'manure_kg': 'Manure',
        'micronutrients_kg': 'Micronutrients',
        'stimulant_kg': 'Growth stimulants',
        'soil_improvement_kg': 'Soil amendments',
        'fuel_MJ': 'Fuel',
        'heat_MJ': 'Heat',
        'electricity_kWh': 'Electricity',
        'yield_kg/ha': 'Crop yield',
        'irrigation_m3': 'Irrigation water',
        'tap_water_kg': 'Process water',
        'protection_kg': 'Crop protection',
        'KG_A' : 'Climatic region',
        'KG_B' : 'Climatic region',
        'KG_C' : 'Climatic region',
        'KG_D' : 'Climatic region',
    }

    color_palette = {}
    for category, predictors_list in predictors.items():
        shades = sns.light_palette(
            base_colors[category], 
            n_colors=len(predictors_list))
        for predictor, shade in zip(reversed(predictors_list), shades):
            color_palette[predictor] = shade  

    plot_order = [pred for group in predictors.values() for pred in group] 

    fig, axes = plt.subplots(
        nrows=len(global_shap_values),
        ncols=1,
        figsize=(15, 1.25 * len(global_shap_values)),
        sharex=True,
        gridspec_kw={'hspace': 0}
    )

    if len(global_shap_values) == 1:
        axes = [axes]

    all_handles = []
    all_labels = []
    plot_orders = []

    for ax, (impact, models) in zip(axes, global_shap_values.items()):
        df_imps = pd.DataFrame(models)
        impact_plot_order = [pred for pred in plot_order if pred in df_imps.index]
        df_imps = df_imps.loc[impact_plot_order]
        df_imps_normalized = (df_imps / df_imps.sum())
        df_imps_normalized = df_imps_normalized.T[impact_plot_order].T

        df_imps_normalized.T.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=[color_palette[col] for col in impact_plot_order]
        )
        df_imps_normalized.to_excel('S8_' + impact.replace(" ", "_") + '_wcat.xlsx')
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
        ax.get_legend().remove()
        ax.tick_params(axis='x', rotation=0)
        ax.set_yticklabels([])
        ax.set_ylabel(
            impact, rotation=0, labelpad=-5, 
            fontsize=20, ha='right', va='center'
            )
        plot_orders.extend(impact_plot_order)

    unique_legend_items = {label: handle for label, handle in zip(all_labels, all_handles) if label in plot_orders}
    sorted_labels = [pred for pred in plot_order if pred in unique_legend_items]
    sorted_handles = [unique_legend_items[label] for label in sorted_labels]
    clean_labels = [label_mapping.get(label, label) for label in sorted_labels]

    fig.legend(
        sorted_handles,
        clean_labels,
        loc='center left',
        bbox_to_anchor=(0.835, 0.5),
        fontsize=20,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(f'predictor_contributions_{p}_new.png', bbox_inches='tight')

# 8) Saving results for the SI
# S2
fig, ax = plt.subplots(1, 1, figsize = (5, 40))
sns.barplot(df_categorical.value_counts('crop'), orient = 'h')
plt.ylabel('Crop species'), plt.xlabel('Number of production activities for crop')
fig.savefig('S2 - Number of activities per crop in the dataset.png')

# S3
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
country_counts = df_categorical['country'].value_counts()
world = gpd.read_file(url)
world.loc[world['NAME'] == 'France', 'ISO_A2'] = 'FR'
world.loc[world['NAME'] == 'Portugal', 'ISO_A2'] = 'PT'
world.loc[world['NAME'] == 'Norway', 'ISO_A2'] = 'NO'
world = world.merge(country_counts.rename('count'), left_on='ISO_A2', right_index=True, how='left')
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
world.plot(column='count', ax=ax, legend=True, cmap='copper', missing_kwds={'color': 'lightgrey'}, vmax = 200)
ax.set_title('Number of rows per country')
fig.savefig('S3 - Map of countries covered in the datasset')
plt.show()

# S6
w = pd.ExcelWriter(f'S6 - optimized_hyperparams_{p}.xlsx')
for impact in impact_order:
    df = pd.DataFrame.from_dict(tuned_estimators_all[impact])
    df.to_excel(w, sheet_name= impact)
w.close()