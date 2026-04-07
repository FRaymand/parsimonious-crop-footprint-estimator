'''
This script:
1) Imports all previously made dataframes of inputs, impacts and cateogircal info
2) Adds information on crop type (annual/perennial) and climatic zone of activity and encodes it
3) Saves the resulting dataframes
4) Generates summary plots
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit
np.random.seed(12)

perspectives = ['H', 'E', 'I']

def calculate_vif(df):
    X = df
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns 
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

perspectives = ['H', 'E', 'I']

for p in perspectives:
    # 1) Load dataframes for each perspective
    df_categorical = pd.read_pickle(f'df_categorical_{p}.pkl')
    df_impacts = pd.read_pickle(f'df_impacts_{p}.pkl')
    df_inputs = pd.read_pickle(f'df_inputs_{p}.pkl')

    # 2) Add crop type and climate zone info
    df_crops = pd.read_excel('C:/Code_Cave/cleaned for review/crop_zone_classification.xlsx', sheet_name='crops')
    df_zones = pd.read_excel('C:/Code_Cave/cleaned for review/crop_zone_classification.xlsx', sheet_name='zones')

    df_categorical = df_categorical.reset_index()

    df_1 = pd.merge(df_categorical, df_crops, left_on='crop', right_on='Crop species')
    df_2 = pd.merge(df_1, df_zones, left_on = 'country', right_on = 'country')
    df_2 = df_2.set_index('activity')

    df_inputs_new = pd.merge(df_inputs, df_2, left_index=True, right_index = True).drop(columns = ['crop', 'country', 'Crop species', 'IMAGE', 'climatic', 'continent', 'crop group'])
    df_inputs_new = pd.concat([df_inputs_new, pd.get_dummies(df_inputs_new[['KG']])], axis=1).drop(columns = ['KG'])
    df_inputs_new = df_inputs_new.astype(float)
    df_inputs = df_inputs_new
    df_inputs = df_inputs.fillna(0)
    y_train_whole = pd.DataFrame(np.log(df_impacts), columns=df_impacts.columns).fillna(0)

    # 3) Save processed data
    with open(f'x_{p}_wcat.pkl', 'wb') as f:
        pickle.dump(df_inputs, f)
    with open(f'y_{p}_wcat.pkl', 'wb') as f:
        pickle.dump(y_train_whole, f)

    # Save summary Excel
    df_categorical.sort_values(by='crop').reset_index().set_index('crop')[['country', 'activity']]\
        .rename(columns={'activity': 'number of activities'})\
        .groupby(['crop', 'country']).count()\
        .to_excel(f'S1_{p}_aug.xlsx')

    # 4) Plots
    sns.set(font_scale=0.85)
    plt.figure(figsize=(5, 20))
    sns.barplot(x=df_categorical['crop'].value_counts().sort_index().values,
                y=df_categorical['crop'].value_counts().sort_index().index)
    plt.ylabel('Crop Species')
    plt.xlabel('Number of production activities for crop')
    plt.tight_layout()
    plt.savefig(f'barplot_crop_counts_{p}_wcat.png')
    plt.close()

    plt.figure(figsize=(5, 20))
    sns.barplot(x=df_categorical['country'].value_counts().sort_index().values,
                y=df_categorical['country'].value_counts().sort_index().index)
    plt.ylabel('Country')
    plt.xlabel('Number of production activities')
    plt.tight_layout()
    plt.savefig(f'barplot_country_counts_{p}_wcat.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_impacts)
    plt.title(f'Boxplot of Impacts - Perspective {p}')
    plt.tight_layout()
    plt.savefig(f'boxplot_impacts_{p}_wcat.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=y_train_whole)
    plt.title(f'Boxplot of log Impacts - Perspective {p}')
    plt.tight_layout()
    plt.savefig(f'boxplot_log_impacts_{p}_wcat.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df_inputs.corr(), vmin=-1, cmap='coolwarm')
    plt.title(f'Pearson Correlation Coefficients - {p}')
    plt.tight_layout()
    plt.savefig(f'heatmap_pearson_{p}_wcat.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df_inputs.corr(method='spearman'), vmin=-1, cmap='coolwarm')
    plt.title(f'Spearman Ranks - {p}')
    plt.tight_layout()
    plt.savefig(f'heatmap_spearman_{p}_wcat.png')
    plt.close()

    # VIF calculation
    vif_df = calculate_vif(df_inputs).sort_values(by='VIF', ascending=False).rename(columns={'VIF': f'VIF_{p}'})
    vif_df.to_excel(f'vif_{p}_wcat.xlsx')

    # Heatmaps for each impact
    x_scaled = df_inputs.astype(float)
    fig, axes = plt.subplots(nrows=1, ncols=len(df_impacts.columns), figsize=(35, 12))
    annot_font_size = 15
    for idx, impact in enumerate(df_impacts.columns):
        df_pearson_feed = x_scaled.merge(
            df_impacts[[impact]], right_index=True, 
            left_index=True, how='inner'
            ).copy()
        sns.heatmap(
            df_pearson_feed.corr(method='spearman')[[impact]].sort_values(by=impact, key=abs),
            cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=axes[idx], 
            annot_kws={"size": annot_font_size}
            )
        axes[idx].set_aspect(aspect='auto')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'heatmap_spearman_impacts_{p}_wcat.png')
    plt.close()

    fig, axes = plt.subplots(
        nrows=1, ncols=len(df_impacts.columns), 
        figsize=(35, 12)
        )
    for idx, impact in enumerate(df_impacts.columns):
        df_pearson_feed = x_scaled.merge(
            df_impacts[[impact]], right_index=True, 
            left_index=True, how='inner'
            ).copy()
        sns.heatmap(
            df_pearson_feed.corr(method='pearson')[[impact]].sort_values(by=impact, key=abs),
            cmap='coolwarm', annot=True, vmin=-1, vmax=1, 
            ax=axes[idx], annot_kws={"size": annot_font_size}
                    )
        axes[idx].set_aspect(aspect='auto')
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'heatmap_pearson_impacts_{p}_wcat.png')
    plt.close()

    # Example scatterplot for yield vs first impact (customize as needed)
    impact_name = df_impacts.columns[0]
    df_pearson_feed = x_scaled.merge(
        df_impacts[[impact_name]], right_index=True, 
        left_index=True, how='inner'
        ).copy()
    plt.figure(figsize=(7, 7))
    plt.xscale('log')
    plt.yscale('log')
    sns.scatterplot(data=df_pearson_feed, x='yield_kg/ha', y=impact_name)
    plt.title(f'Scatterplot yield vs {impact_name} - {p}')
    plt.tight_layout()
    plt.savefig(f'scatter_yield_vs_{impact_name}_{p}_wcat.png')
    plt.close()