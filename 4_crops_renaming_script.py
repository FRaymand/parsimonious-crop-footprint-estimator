'''
This script:
1) imports all previously made dataframes of inputs, impacts and cateogircal info 
2) Ensures the various dataframes correspond and filters irrelevant activities
3) Manually fixes the names of various crop production activities
4) Writes the finalized dataframes to be used in analysis
'''

import pandas as pd
import numpy as np
import pickle
import copy
np.random.seed(12)

def merge_dicts_double(dict1, dict2):
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
    if not dict_list:
        return {}
    merged_dict = dict_list[0]
    for d in dict_list[1:]:
        merged_dict = merge_dicts_double(merged_dict, d)
    
    return merged_dict


def process_perspective(perspective):
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
    print('Section 1 Done')

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
    print('Section 2 Done')

    # 3) Manual crop renaming + Duplicate Removal
    df_categorical.loc[['codes for activitiesneeding manual renaming'], 'crop'] = 'crop to be renamed to' # ree=peat as necessary
 
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['triticale grain', 'Triticale grain'])), 'crop'] = 'triticale'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Forage barley', 'forage barley'])), 'crop'] = 'barley'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['soybeans', 'soybeans'])), 'crop'] = 'soybean'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Fodder', 'fodder'])), 'crop'] = 'fodder beet'
    df_categorical.loc[df_categorical['crop'].str.contains('|'.join(['Field bean', 'field bean', 'field'])), 'crop'] = 'faba bean'
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
    print('Section 3 Done')

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
    print(f'Processing for perspective {perspective} done.')


if __name__ == "__main__":
    for perspective in ['H', 'E', 'I']:
        process_perspective(perspective)
    print('All perspectives processed.')