'''
Processing Script: Calculating best predictor subset

Input Specifications:
    - Pickled files of LCI and footprints of crop production.
        - Type: Pandas Dataframe

Output Specifications:
    - Dictionary of best predictor subsets for each combination of (impact, model, size)
        - Type: Pickled Dictionary

Script process:
    1) Imports LCI and footprint dataframes, and defines parameters (for each perspective)
    2) Derives the best predictor subset for each (impact x model x predictor number)
        and pickles it.

Prerequisite:
    - 4_preprocessing.py

Prerequisite for:
    - 6_model_hyperparamTuning.py
'''

import pickle
import logging
import itertools
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, PowerTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from optuna.trial import Trial
np.random.seed(12)


def get_feature_combinations(n_features: int, subset_size: int):
    '''extracts all possible combinations of features for a given subset_size'''
    combinations = list(itertools.combinations(range(n_features), subset_size))
    return [",".join(map(str, comb)) for comb in combinations]

def objective(trial: Trial) -> float:
    '''
    The function being optimized by Optuna. Here we set this to minimize RMSE for
    a given subset size, modelling technique, and impact. The subset of predictors
    that lead to the minimal RMSE are therefore our best predictor subset.
    
    '''
    n_features = x_train.shape[1] - x_train.columns.str.contains('KG_').sum() + 1
    feature_combinations = get_feature_combinations(n_features, subset_size)
    if not feature_combinations:
        return float("-inf")

    selected_features_str = trial.suggest_categorical("selected_subset", feature_combinations)
    selected_features = tuple(map(int, selected_features_str.split(",")))
    x_train_subset = x_train.iloc[:, list(selected_features)]

    if x_train_subset.columns.str.contains('KG_').sum() > 0: # Add rest of climatic info
        df_addition = x_train.iloc[:, x_train.columns.str.contains('KG_')]
        cols_to_use = df_addition.columns.difference(x_train_subset.columns)
        x_train_subset = pd.merge(x_train_subset, 
                                  df_addition[cols_to_use], 
                                  left_index=True, 
                                  right_index=True, 
                                  how='outer')
    classifier_obj = models[estimator_name]

    pipeline = Pipeline([
        ('transformer', transformer),
        ('scaler', MinMaxScaler((0, 1))),
        ('regressor', classifier_obj) 
        ])
            
    score = cross_validate(
        pipeline, x_train_subset, y_train_impact, cv=cv, 
        groups = groups_encoded,  scoring='neg_root_mean_squared_error',  
        n_jobs=-1, return_train_score=False
        )['test_score'].mean()
    return score

# 1) Imports and definitions
perspective_list = ['H', 'E', 'I']
models = {'RF' : RandomForestRegressor(n_jobs = 1, random_state=12),
            'KNN' : KNeighborsRegressor(n_jobs=1),
            'ANN' : MLPRegressor(hidden_layer_sizes=[10], random_state=12, max_iter=3500),
            'GBM' : HistGradientBoostingRegressor(random_state = 12), 
            'GLM' : Ridge() }    
enc = OrdinalEncoder()
transformer = PowerTransformer()
cv = GroupKFold(n_splits = 5, shuffle = True, random_state = 12)
list_subset_sizes = list(np.arange(1,17))

for perspective in perspective_list:
    logging.info(f'Reading input files for {perspective} perspective ...')
    x_train = pd.read_pickle(f'x_{perspective}_wcat.pkl')
    y_train = pd.read_pickle(f'y_{perspective}_wcat.pkl')
    df_impacts = pd.read_pickle(f'df_impacts_{perspective}.pkl')
    df_categorical = pd.read_pickle(f'df_categorical_{perspective}.pkl')
     
    best_estimators = {key: {model:{} for model in models.keys()} for key in df_impacts.columns}
    estimators = models.keys()
    groups = df_categorical.loc[x_train.index][['crop']]
    groups_encoded = enc.fit_transform(groups)

    # 2) Find best subset of predictors and save to (pickle) file
    for impact in df_impacts.columns:
        y_train_impact = y_train[impact].copy()

        for estimator_name in estimators:
            best_estimators[impact] = {estimator_name: {}}
            for subset_size in list_subset_sizes:

                search_space = {"selected_subset": get_feature_combinations(
                    x_train.shape[1] - x_train.columns.str.contains('KG_').sum() + 1,
                    subset_size
                    )}

                if not search_space["selected_subset"]:
                    raise RuntimeError(f"No valid combinations found for subset size: {subset_size}.")

                sampler = optuna.samplers.GridSampler(search_space)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(objective, n_trials=1500, n_jobs=-1)
                if len(study.trials) == 0 or study.best_trial.state != optuna.trial.TrialState.COMPLETE:
                    logging.warning(f"No completed trials for subset size {subset_size}. Skipping.")
                    continue

                # Store best subset
                best_subset = tuple(map(int, study.best_params["selected_subset"].split(",")))
                best_estimators[impact][estimator_name][subset_size] = best_subset
                logging.info(f"Best subset for {estimator_name} at size {subset_size}: {best_subset}")

        with open(f'dict_predictor_importances_{perspective}_{impact}_{estimator_name}.pkl', 'wb') as f:
            pickle.dump(best_estimators, f)