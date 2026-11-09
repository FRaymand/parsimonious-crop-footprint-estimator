'''
Processing Script: Preprocessing model inputs

Input Specifications:
    - Pickled dataframes of input LCI and footprints from all databases

Output Specifications:
    - Pickled dataframes of input LCI and footprints after processing

Script process:
    1) imports all previously made dataframes of inputs, impacts and cateogircal info 
    2) Ensures the various dataframes correspond and filters irrelevant activities
    3) Manually fixes the names of various crop production activities
    4) Writes the finalized dataframes to be used in analysis
    5) Imports all previously made dataframes of inputs, impacts and cateogircal info
    6) Adds information on crop type (annual/perennial) and climatic zone of activity and encodes it
    7) Saves the resulting dataframes
    8) Generates summary plots

Prerequisite:
    - 1_dataCompilation_ecoinvent.py
    - 2_dataCompilation_wfldb.py
    - 3_dataCompilation_agribalyse.py

Prerequisite for:
    - 5_model_predictorSelection.py
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
np.random.seed(12)


def calculate_vif(df):
    X = df
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns 
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data


perspectives = ['H', 'E', 'I']

for perspective in perspectives:

    # 1) Load dataframes
    df_inputs_ei = pd.read_pickle('df_ecoinvent_inputs_3.11.pkl').rename(columns={'yield': 'yield_kg/ha'})
    df_categorical_ei = pd.read_pickle('df_ecoinvent_categorical_3.11.pkl')
    df_impacts_ei = pd.read_pickle(f'df_impacts_ecoinvent_3_11_{perspective}.pkl') \
        .drop(columns=['crop', 'country']) \
        .rename(columns={'PDF_noLT': 'Biodiversity Loss', 'WU_noLT': 'Water Use', 'CC_noLT': 'Climate Change', 'code': 'activity'}) \
        .set_index('activity').sort_values(by='activity')

    df_inputs_ag = pd.read_pickle('df_agb_inputs.pkl')
    df_categorical_ag = pd.read_pickle('df_agb_categorical.pkl')
    df_impacts_ag = pd.read_pickle(f'df_agb_impacts_{perspective}.pkl').rename(
        columns={'Ecosystems': 'Biodiversity Loss', 'Water consumption': 'Water Use', 'Global warming': 'Climate Change', 'code': 'activity'})

    df_inputs_wf = pd.read_pickle('df_wfldb_inputs.pkl')
    df_categorical_wf = pd.read_pickle('df_wfldb_categorical.pkl')
    df_impacts_wf = pd.read_pickle(f'df_wfldb_impacts_{perspective}.pkl').rename(
        columns={'Ecosystems': 'Biodiversity Loss', 'Water consumption': 'Water Use', 'Global warming': 'Climate Change', 'code': 'activity'})
    logging.info('Section 1 Done')

    # 2) Concatenate and clean
    df_categorical = pd.concat([df_categorical_ei, df_categorical_ag, df_categorical_wf])
    df_categorical = df_categorical[~df_categorical['country'].isin(['GLO', 'RER', 'RoW'])].copy()
    df_categorical = df_categorical.rename(columns={'code': 'activity'}).set_index('activity').sort_values(by='activity')
    df_categorical = df_categorical.loc[np.sort(df_categorical.index)]

    df_inputs = pd.concat([df_inputs_ei, df_inputs_ag, df_inputs_wf])
    df_inputs = df_inputs.loc[np.sort(df_inputs.index)]

    df_impacts = pd.concat([df_impacts_ei, df_impacts_ag, df_impacts_wf]).astype(float)
    df_impacts = df_impacts.loc[np.sort(df_impacts.index)]

    common_indices = np.intersect1d(np.intersect1d(df_inputs.index, df_impacts.index), df_categorical.index)
    df_inputs = df_inputs.loc[common_indices]
    df_impacts = df_impacts.loc[common_indices]
    df_categorical = df_categorical.loc[common_indices]
    df_impacts = df_impacts.loc[df_impacts[(df_impacts > 0).all(axis=1)].index]
    logging.info('Section 2 Done')

    # 3) Manual crop renaming + Duplicate Removal
    df_categorical.loc[['codes for activitiesneeding manual renaming'], 'crop'] = 'crop to be renamed to' # ree=peat as necessary

    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['triticale grain', 'Triticale grain'])), 'crop'] = 'triticale'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Forage barley', 'forage barley'])), 'crop'] = 'barley'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['soybeans', 'soybeans'])), 'crop'] = 'soybean'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Fodder', 'fodder'])), 'crop'] = 'fodder beet'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Field bean', 'field bean'])), 'crop'] = 'faba bean'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Cocoa', 'cocoa'])), 'crop'] = 'cocoa'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Tomato', 'tomato'])), 'crop'] = 'tomato'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Almond', 'almond'])), 'crop'] = 'almond'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Strawberry', 'strawberry'])), 'crop'] = 'strawberry'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Palm fruit', 'palm fruit'])), 'crop'] = 'palm fruit bunch'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Apple', 'Cider apple', 'apple'])), 'crop'] = 'apple'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Banana'])), 'crop'] = 'banana'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Barley grain', 'Spring barley', 'Winter barley'])), 'crop'] = 'barley'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Winter forage barley'])), 'crop'] = 'forage barley'
    df_categorical.loc[df_categorical['crop']== 'Barley', 'crop'] = 'barley'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Beetroot'])), 'crop'] = 'beet root'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Sugar beet'])), 'crop'] = 'sugar beet'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['lupine', 'lupin', 'Lupin'])), 'crop'] = 'lupine'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Brazil'])), 'crop'] = 'brazil nuts'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Cashew', 'cashew'])), 'crop'] = 'cashew'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Chick', 'chickpea', 'Chickpea'])), 'crop'] = 'chickpea'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Chicory root', 'Chicory roots'])), 'crop'] = 'chicory root'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Chilli', 'chilli'])), 'crop'] = 'chilli'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Coconut', 'coconut'])), 'crop'] = 'coconut'
    df_categorical.loc[df_categorical['crop']== 'Coffee, cherries (Arabica)', 'crop'] = 'coffee arabica'
    df_categorical.loc[df_categorical['crop']== 'Coffee, cherries (Robusta)', 'crop'] = 'coffee robusta'
    df_categorical.loc[df_categorical['crop']== 'Coffee bean (Robusta)', 'crop'] = 'coffee robusta'
    df_categorical.loc[df_categorical['crop']== 'Coffee, green beans (Robusta)', 'crop'] = 'coffee robusta'
    df_categorical.loc[df_categorical['crop']== 'coffee', 'crop'] = 'coffee arabica'
    df_categorical.loc[df_categorical['crop']== 'Coffee, green beans (Arabica)', 'crop'] = 'coffee arabica'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Coriander'])), 'crop'] = 'coriander'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Cotton'])), 'crop'] = 'cotton'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Stevia'])), 'crop'] = 'stevia'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Pomegranate'])), 'crop'] = 'pomegranate'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Spring pea', 'Winter pea', 'Dry pea'])), 'crop'] = 'dry pea'
    df_categorical.loc[df_categorical['crop']== 'pea', 'crop'] = 'green pea'
    df_categorical.loc[df_categorical['crop']== 'Pea', 'crop'] = 'green pea'
    df_categorical.loc[df_categorical['crop']== 'Green pea', 'crop'] = 'green pea'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Garlic'])), 'crop'] = 'garlic'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Ginger'])), 'crop'] = 'ginger'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Wine grape', 'wine grape'])), 'crop'] = 'wine grape'
    df_categorical.loc[df_categorical['crop']== 'Grape', 'crop'] = 'wine grape'
    df_categorical.loc[df_categorical['crop']== 'grape', 'crop'] = 'table grape'
    df_categorical.loc[df_categorical['crop']== 'Grape, table grape', 'crop'] = 'table grape'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Grass', 'grass'])), 'crop'] = 'grass'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Hibiscus'])), 'crop'] = 'hibiscus'
    df_categorical.loc[df_categorical['crop']== 'fava', 'crop'] = 'faba bean'
    df_categorical.loc[df_categorical['crop']== 'Winter faba bean', 'crop'] = 'faba bean'
    df_categorical.loc[df_categorical['crop']== 'Spring faba bean', 'crop'] = 'faba bean'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['rice', 'Rice'])), 'crop'] = 'rice'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Kiwi', 'kiwi'])), 'crop'] = 'kiwi'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Lentils', 'lentil'])), 'crop'] = 'lentil' # the first type is also dried
    df_categorical.loc[df_categorical['crop']== 'Maize grain, irrigated', 'crop'] = 'maize'
    df_categorical.loc[df_categorical['crop']== 'Maize grain, non-irrigated', 'crop'] = 'maize'
    df_categorical.loc[df_categorical['crop']== 'Maize', 'crop'] = 'maize'
    df_categorical.loc[df_categorical['crop']== 'Maize grain', 'crop'] = 'maize'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Mandarin'])), 'crop'] = 'mandarin'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Mango'])), 'crop'] = 'mango'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Oat', 'oat'])), 'crop'] = 'oat'
    df_categorical = df_categorical[~df_categorical['crop'].str.contains('|'.join(['Onion sets']))]
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Onion'])), 'crop'] = 'onion'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Orange'])), 'crop'] = 'orange'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Peach'])), 'crop'] = 'peach'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Peanut'])), 'crop'] = 'peanut'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Pear'])), 'crop'] = 'pear'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Rapeseed', 'rapeseed'])), 'crop'] = 'rapeseed'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Quinoa'])), 'crop'] = 'quinoa'
    df_categorical = df_categorical[~df_categorical['crop'].str.contains('|'.join(['Rose']))]
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Rosemary'])), 'crop'] = 'rosemary'
    df_categorical = df_categorical[~df_categorical['crop'].str.contains('|'.join(['Silage sorghum']))]
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Sorghum'])), 'crop'] = 'sorghum'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Soybean'])), 'crop'] = 'soybean'
    df_categorical.loc[df_categorical['crop']== 'palm', 'crop'] = 'palm fruit bunch'
    df_categorical.loc[df_categorical['crop']== 'bell', 'crop'] = 'bell pepper'
    df_categorical.loc[df_categorical['crop']== 'black pepper bells', 'crop'] = 'bell pepper'
    df_categorical.loc[df_categorical['crop']== 'red', 'crop'] = 'red kidney bean'
    df_categorical.loc[df_categorical['crop']== 'navy', 'crop'] = 'navy bean'
    df_categorical.loc[df_categorical['crop']== 'castor', 'crop'] = 'castor bean'
    df_categorical.loc[df_categorical['crop']== 'pinto', 'crop'] = 'pinto bean'
    df_categorical.loc[df_categorical['crop']== 'white', 'crop'] = 'white asparagus'
    df_categorical.loc[df_categorical['crop']== 'sunn', 'crop'] = 'sunnhemp'
    df_categorical.loc[df_categorical['crop']== 'hemp', 'crop'] = 'sunnhemp'
    df_categorical.loc[df_categorical['crop']== 'Sweet corn', 'crop'] = 'maize'
    df_categorical.loc[df_categorical['crop']== 'Ware potato', 'crop'] = 'potato'
    df_categorical.loc[df_categorical['crop']== 'Starch potato', 'crop'] = 'potato'
    df_categorical.loc[df_categorical['crop']== 'Sunflower grain', 'crop'] = 'sunflower'
    df_categorical.loc[df_categorical['crop']== 'Durum wheat grain', 'crop'] = 'durum wheat grain'
    df_categorical.loc[df_categorical['crop']== 'Soft wheat grain', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop']== 'Wheat grain', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop']== 'Wheat grain, irrigated', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop']== 'Wheat grain, non-irrigated', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop']== 'Winter wheat', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop']== 'wheat', 'crop'] = 'wheat grain'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Spinach'])), 'crop'] = 'spinach'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Sugarcane'])), 'crop'] = 'sugarcane'
    df_categorical.loc[df_categorical['crop']== 'sugar', 'crop'] = 'sugar beet'
    df_categorical.loc[df_categorical['crop']== 'protein', 'crop'] = 'protein pea'
    df_categorical.loc[df_categorical['crop']== 'Black pepper bells', 'crop'] = 'bell pepper'
    df_categorical.loc[df_categorical['crop']== 'Pumpkin, organic', 'crop'] = 'pumpkin'
    df_categorical.loc[df_categorical['crop']== 'Pumpkin', 'crop'] = 'pumpkin'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Vanilla'])), 'crop'] = 'vanilla'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Walnut'])), 'crop'] = 'walnut'
    df_categorical.loc[df_categorical['crop']== 'Aloe vera leaves', 'crop'] = 'aloe vera'
    df_categorical.loc[df_categorical['crop']== 'Grapefruit for juice', 'crop'] = 'grapefruit'
    df_categorical.loc[df_categorical['crop']== 'Hazelnut, in shell', 'crop'] = 'hazelnut'
    df_categorical.loc[df_categorical['crop']== 'Macadamia nut, in shell', 'crop'] = 'macadamia nut'
    df_categorical.loc[df_categorical['crop']== 'Pistachio, in shell', 'crop'] = 'pistachio'
    df_categorical.loc[df_categorical['crop']== 'Shea fruit, Sahel region', 'crop'] = 'shea fruit'
    df_categorical.loc[df_categorical['crop']== 'Tea, fresh leaves', 'crop'] = 'tea'
    logging.info('Section 3 Done')

    # 4) Deleting dupicated activities identified across the databases
    list_duplicates = ['DUPLICATE ACTIVITIES CODES'] # Agribalyse activities
    list_preprocessings = ['DUPLICATE ACTIVITIES CODES'] # WFLDB activities

    df_categorical = df_categorical.loc[~df_categorical.index.isin(list_preprocessings + list_duplicates)]    
    df_categorical['crop'] = df_categorical['crop'].str.lower()

    common_indices = np.intersect1d(np.intersect1d(df_inputs.index, df_impacts.index), df_categorical.index)
    df_inputs = df_inputs.loc[common_indices]
    df_impacts = df_impacts.loc[common_indices]
    df_categorical = df_categorical.loc[common_indices]

    # Save outputs with perspective in filename
    with open(f'df_inputs_{perspective}.pkl', 'wb') as f:
        pickle.dump(df_inputs, f)
    with open(f'df_categorical_{perspective}.pkl', 'wb') as f:
        pickle.dump(df_categorical, f)
    with open(f'df_impacts_{perspective}.pkl', 'wb') as f:
        pickle.dump(df_impacts, f)
    logging.info(f'Processing for perspective {perspective} done.')



    # 5) Load dataframes for each perspective
    df_categorical = pd.read_pickle(f'df_categorical_{perspectives}.pkl')
    df_impacts = pd.read_pickle(f'df_impacts_{perspectives}.pkl')
    df_inputs = pd.read_pickle(f'df_inputs_{perspectives}.pkl')

    # 6) Add crop type and climate zone info
    df_crops = pd.read_excel('crop_zone_classification.xlsx', sheet_name='crops')
    df_zones = pd.read_excel('crop_zone_classification.xlsx', sheet_name='zones')

    df_categorical = df_categorical.reset_index()

    df_1 = pd.merge(df_categorical, df_crops, left_on='crop', right_on='Crop species')
    df_2 = pd.merge(df_1, df_zones, left_on = 'country', right_on = 'country')
    df_2 = df_2.set_index('activity')

    df_inputs_new = pd.merge(df_inputs, 
                             df_2, 
                             left_index=True, 
                             right_index = True).drop(columns = ['crop', 'country', 'Crop species', 
                                                                 'IMAGE', 'climatic', 'continent', 
                                                                 'crop group'])
    df_inputs_new = pd.concat([df_inputs_new, pd.get_dummies(df_inputs_new[['KG']])], axis=1).drop(columns = ['KG'])
    df_inputs_new = df_inputs_new.astype(float)
    df_inputs = df_inputs_new
    df_inputs = df_inputs.fillna(0)
    y_train_whole = pd.DataFrame(np.log(df_impacts), columns=df_impacts.columns).fillna(0)

    # 7) Save processed data
    with open(f'x_{perspectives}_wcat.pkl', 'wb') as f:
        pickle.dump(df_inputs, f)
    with open(f'y_{perspectives}_wcat.pkl', 'wb') as f:
        pickle.dump(y_train_whole, f)

    # Save summary Excel
    df_categorical.sort_values(by='crop').reset_index().set_index('crop')[['country', 'activity']]\
        .rename(columns={'activity': 'number of activities'})\
        .groupby(['crop', 'country']).count()\
        .to_excel(f'S1_{perspectives}_aug.xlsx')

    # 8) Plots
    sns.set(font_scale=0.85)
    plt.figure(figsize=(5, 20))
    sns.barplot(x=df_categorical['crop'].value_counts().sort_index().values,
                y=df_categorical['crop'].value_counts().sort_index().index)
    plt.ylabel('Crop Species')
    plt.xlabel('Number of production activities for crop')
    plt.tight_layout()
    plt.savefig(f'barplot_crop_counts_{perspectives}_wcat.png')
    plt.close()

    plt.figure(figsize=(5, 20))
    sns.barplot(x=df_categorical['country'].value_counts().sort_index().values,
                y=df_categorical['country'].value_counts().sort_index().index)
    plt.ylabel('Country')
    plt.xlabel('Number of production activities')
    plt.tight_layout()
    plt.savefig(f'barplot_country_counts_{perspectives}_wcat.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_impacts)
    plt.title(f'Boxplot of Impacts - Perspective {perspectives}')
    plt.tight_layout()
    plt.savefig(f'boxplot_impacts_{perspectives}_wcat.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=y_train_whole)
    plt.title(f'Boxplot of log Impacts - Perspective {perspectives}')
    plt.tight_layout()
    plt.savefig(f'boxplot_log_impacts_{perspectives}_wcat.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df_inputs.corr(), vmin=-1, cmap='coolwarm')
    plt.title(f'Pearson Correlation Coefficients - {perspectives}')
    plt.tight_layout()
    plt.savefig(f'heatmap_pearson_{perspectives}_wcat.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df_inputs.corr(method='spearman'), vmin=-1, cmap='coolwarm')
    plt.title(f'Spearman Ranks - {perspectives}')
    plt.tight_layout()
    plt.savefig(f'heatmap_spearman_{perspectives}_wcat.png')
    plt.close()

    # VIF calculation
    vif_df = calculate_vif(df_inputs).sort_values(by='VIF', ascending=False).rename(columns={'VIF': f'VIF_{perspectives}'})
    vif_df.to_excel(f'vif_{perspectives}_wcat.xlsx')

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
    plt.savefig(f'heatmap_spearman_impacts_{perspectives}_wcat.png')
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
    plt.savefig(f'heatmap_pearson_impacts_{perspectives}_wcat.png')
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
    plt.title(f'Scatterplot yield vs {impact_name} - {perspectives}')
    plt.tight_layout()
    plt.savefig(f'scatter_yield_vs_{impact_name}_{perspectives}_wcat.png')
    plt.close()