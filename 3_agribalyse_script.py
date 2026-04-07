'''
This script:
1) Creates a new brightway project, imports Agribalyse
2) Makes a database of activities in agribalyse with misc. units, since 
        these could be used as background activities.
3) Filters activities to collect crop production LCIs
4) Converts units and harmonizes unit spellings
5) Converts fertiliser inputs to active ingredients (NPK) & aggregates predictors
6) Converts units and harmonizes unit spellings, second pass
7) Constructs df_inputs
8) Constructs df_categorical
9) Constructs impacts dataframe for all three perspectives
'''

import pandas as pd
import numpy as np
import sympy
import brightway2 as bw
import bw2io as bi
import glob
import pickle
from tqdm import tqdm


def find_value_for_keyword(df, keyword):
    if df.loc[df['column1'] == keyword , 'column2'].values[0]:
        value = df.loc[df['column1'] == keyword , 'column2'].values[0]
    return value


def find_products(df):
    ''' used to find the name of product(s) '''
    i, j = (df[df['column1'].isin(['Products','Avoided products'])].index)
    df = df.loc[i+1:j-1, ['column1', 'column3']].dropna(axis = 0, how = 'all')
    return {'name' : df['column1'].values, 'unit' : df['column3'].values}


def find_normalization_value(df, name):
    return df.loc[df.index[df['column1'] == 'Products'] + 1, 'column2'].iloc[0]


def find_product_unit(df, name):
    return df.loc[df.index[df['column1'] == 'Products'] + 1, 'column3'].iloc[0]


def get_biosphere(df, normalization_factor):
    i, j = (df[df['column1'].isin(['Resources','Materials/fuels'])].index)
    df_bio = df.loc[i+1:j-1, ['column1', 'column2', 'column3', 'column4', 'column5','column6', 'column9']].dropna(axis = 0, how = 'all').dropna(axis = 0, subset = 'column3')
    df_bio.columns = ['new_input','medium',  'amount', 'unit', 'distribution', 'sd', 'description']
    df_bio['new_input'] = [name.split('{')[0] for name in df_bio['new_input']]
    df_bio['amount'] = [float(sympy.sympify(field)) if type(field) != float else field for field in df_bio['amount']]
    df_bio['amount'] = df_bio['amount'] / normalization_factor
    df_bio['code'] = code 
    df_bio['activity'] = name
    df_bio['location'] = location
    df_bio = df_bio.drop(columns = ['medium', 'distribution', 'sd', 'description'])

    return df_bio


def get_technosphere(df, normalization_factor):
    i, j = (df[df['column1'].isin(['Materials/fuels', 'Emissions to air'])].index)
    df_techno = df.loc[i+1:j-1, ['column1', 'column2', 'column3', 'column4', 'column5',  'column8']].dropna(axis = 0, how = 'all').dropna(axis = 0, subset = 'column2')
    df_techno.columns = ['new_input', 'amount', 'unit', 'distribution', 'sd', 'description']
    df_techno['amount'] = [float(sympy.sympify(field)) if type(field) != float else field for field in df_techno['amount']]
    df_techno['amount'] = df_techno.loc[:, 'amount'] / normalization_factor
    df_techno['code'] = code 
    df_techno['activity'] = name
    df_techno = df_techno.drop(columns = ['distribution', 'sd', 'description'])
    df_techno['location'] = location 
    return df_techno


def get_unique_exchanges(unit, df):
    return print(*df[df['unit'] == unit]['exchange'].unique(), sep = '\n')


def add_bucket(keyword, unit, subbucket, bucket, df):
    '''keywords could be a list and it will still look for all of them'''
    if type(keyword) != list:
        keywords = [keyword, keyword.capitalize()]
    else:
        keywords = [word.capitalize() for word in keyword] + keyword
    df.loc[(df['exchange'].str.contains('|'.join(keywords))) & (df['unit'] == unit), 'bucket'] = bucket
    df.loc[(df['exchange'].str.contains('|'.join(keywords))) & (df['unit'] == unit), 'subbucket'] = subbucket


def is_market_mix(activity):
    """
    Determines if an activity is a market mix by checking if:
    - All technosphere inputs have the same name/unit.
    - The inputs come from multiple regions.
    - The activity name contains 'market for'.
    """
    technosphere_inputs = [
        exc.input for exc in activity.exchanges() if exc["type"] == "technosphere"
    ]
    
    if len(technosphere_inputs) < 2:
        return False  # Not enough inputs to be a mix

    # Check if all inputs share the same name and unit but differ in location
    input_names = {inp["name"] for inp in technosphere_inputs}
    input_units = {inp["unit"] for inp in technosphere_inputs}
    input_locations = {inp.get("location", None) for inp in technosphere_inputs}

    if len(input_names) == 1 and len(input_units) == 1 and len(input_locations) > 1:
        return True  # Likely a market mix
    
    # Additional check: If name contains "market for", it is very likely a mix
    if "market for" in activity["name"].lower():
        return True

    return False

def find_valid_inputs_ei(activity, parent_amount = 1, visited=None):
    """
    Recursively finds technosphere inputs that are in kg or MJ.
    If an input has another unit and is a market mix, it goes one level deeper.
    """
    if visited is None:
        visited = set()
    if activity["code"] in visited:  # Prevent infinite loops
        return []
    visited.add(activity["code"])
    valid_inputs = []
    
    for exc in activity.exchanges():
        if exc["type"] == "technosphere":
            input_activity = exc.input
            input_unit = input_activity["unit"]
            adjusted_amount = exc.amount * parent_amount
            
            if 'packaging' in input_activity['name']:
                continue  # Skip this iteration if 'packaging' is found
            if (input_unit in ['cubic meter', 'm3']) and ('liquid manure spreading' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif (input_unit in ['litre', 'l']) and ('drying' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif input_unit in ['kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit']:
                valid_inputs.append((exc, adjusted_amount))
            elif (is_market_mix(input_activity)) and (input_unit not in {'kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit', 'litre'}):  
                # Dive deeper if it's a market mix AND not kg/MJ
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif (input_unit in ['hectare', 'Ha', 'ha']) and ('trellis system' not in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))
            elif input_unit == 'hour':
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))
            elif (input_unit in ['ton kilometer', 'tkm']) and ('transport, tractor and trailer, agricultural' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))

    return valid_inputs


def find_valid_inputs_agb(activity, parent_amount = 1, visited=None):
    """
    Recursively finds technosphere inputs that are in kg or MJ.
    If an input has another unit and is a market mix, it goes one level deeper.
    """
    if visited is None:
        visited = set()
    if activity in visited:  # Prevent infinite loops
        return []
    visited.add(activity)

    valid_inputs = []
    j = dict_agb_miscunits_acts.get(activity)
    df_process = df_agb_miscunits.loc[j:process_start_index_miscunits[process_start_index_miscunits.index(j)+1]-1].copy()
    code = find_value_for_keyword(df_process, 'Process identifier')
    name = find_value_for_keyword(df_process, 'Process name')

    if type(name) != str:
        name = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
        
    normalization_factor = find_normalization_value(df_process, name)
    unit = find_product_unit(df_process, name)
    if parent_amount != 0:
        df_techno = get_technosphere(df_process, normalization_factor/ parent_amount)
    else:
        return []

    if len (df_techno) == 0:
        print('uh oh!', name, ' has no technosphere? from activity')
    df_techno = df_techno[~df_techno['unit'].isin(['m2', 'm2a', np.nan])]
    valid_inputs.append(df_techno[df_techno['unit'].isin(['kg', 'ton', 'kWh', 'l', 'm3', 'MJ', 'kilogram', 'litre', 'cubic meter', 'megajoule'])])
    df_techno = df_techno[~df_techno['unit'].isin(['kg', 'ton', 'kWh', 'l', 'm3', 'MJ', 'kilogram', 'litre', 'cubic meter', 'megajoule'])]

    for index, row in df_techno.iterrows():
        input_activity = row['new_input']
        adjusted_amount = row['amount']
        
        if ('Ecoinvent' in row['new_input']):
            list_data = []
            name_exc = row['new_input']
            
            if ('market group' in name_exc) and ('Transport,' not in name_exc):
                activity_type = 'market group'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[1][1:-1]) == str.lower(act['name'])) and  (act['activity type'] == activity_type)]

            elif ('market for' in name_exc) and ('Transport,' not in name_exc):
                activity_type = 'market activity'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[1][1:-1]) == str.lower(act['name'])) and  (act['activity type'] == activity_type)]
                if len(list_acts_in_eco) == 0:
                    list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[0].split('{')[0][:-1]) in str.lower(act['name'])) and  (act['activity type'] == activity_type)]

            elif 'Transport, ' in name_exc:
                continue

            else:
                activity_type = 'ordinary transforming activity'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[0].split('{')[0][:-1]) in str.lower(act['name']))]

            if len(list_acts_in_eco) == 1:
                exc_to_update = list_acts_in_eco[0]
                updated_input = find_valid_inputs_ei(exc_to_update, row['amount'])

                for exc, adjusted_amount in updated_input:
                    list_data.append({  "activity": name, 'location': location, 'code' : code, "new_input": exc.input["name"],"amount": adjusted_amount, "unit": exc.input["unit"]   })

            valid_inputs.append(pd.DataFrame(list_data))

        else:
            list_data_agb = []
            if (row['unit'] in ['cubic meter', 'm3']) and ('Irrigating,' in row['new_input']):
                valid_inputs.extend(find_valid_inputs_agb(input_activity, adjusted_amount))  
                continue

            elif (row['unit'] == 'ha') and ('trellis system' not in row['new_input']):
                valid_inputs.extend(find_valid_inputs_agb(input_activity, adjusted_amount))
                continue

            elif (row['unit'] in {'litre', 'l'}) and ('drying' in row['new_input']): # i want the electricity and/or heat inputs not the flow of drying (which is the removal of water)
                valid_inputs.extend(find_valid_inputs_agb(input_activity, adjusted_amount))
                continue

            elif (row['unit'] in ['kg', 'ton', 'kWh', 'l', 'MJ', 'kilogram', 'litre', 'megajoule', 'm3']):
                list_data_agb.append({"activity": name,'location': location,'code' : code, "new_input":input_activity, "amount": adjusted_amount, "unit": row["unit"] })
                valid_inputs.append(pd.DataFrame(list_data_agb))

            else:
                valid_inputs_interrim = find_valid_inputs_agb(input_activity, adjusted_amount, visited)
                if type(valid_inputs_interrim) != list :
                    valid_inputs.append()

    return valid_inputs


def one_level_deeper_techno(df, name, code):
    updated_activities = []
    updated_activities.append(df.loc[ (df['unit'].isin(['kg', 'ton', 'kWh', 'l', 'MJ', 'kilogram', 'litre', 'megajoule'])) & (~df['new_input'].str.contains('Harvesting, with balling') )])
    updated_activities.append(df[df['unit'].isin(['m3', 'cubic meter']) & ('vacuum tanker' not in df['new_input'])])

    df = df[~(df['unit'].isin(['kg', 'ton', 'kWh', 'l', 'MJ', 'kilogram', 'litre', 'megajoule', 'm2', 'm2a', np.nan]) & (~df['new_input'].str.contains('Harvesting, with balling')))] 
    df = df[~((df['unit'].isin(['cubic meter', 'm3'])) & (~df['new_input'].str.contains('Liquid manure spreading') ) & (~df['new_input'].str.contains('vacuum tanker') ))]
    df = df[~(df['new_input'].str.contains('|'.join(['seed', 'Seed', 'Seedling', 'seedling', 'jasmine rice','winter wheat', 'spring barley', 'chickpea', 'straw', 'grain', 'winter', 'spring', 'apple', 'Apple', 'Soybean', 'Starch', 'blue lupine',
                                                'Jasmine rice', 'Chickpea', 'Straw', 'Grain', 'Winter', 'Spring', 'apple', 'Apple', 'Soybean', 'Starch', 'Blue lupine', 'transport, ', 'Transport, '])))]

    for index, row in df.iterrows():

        if 'ecoinvent' in str.lower(row['new_input']):
            list_data = []
            name_exc = row['new_input']
            if ('market group' in name_exc): 
                activity_type = 'market group'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[1][1:-1]) == str.lower(act['name'])) and  (act['activity type'] == activity_type)]

            elif ('market for' in name_exc):
                activity_type = 'market activity'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[1][1:-1]) == str.lower(act['name'])) and  (act['activity type'] == activity_type)]
                if len(list_acts_in_eco) == 0:
                    list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[0].split('{')[0][:-1]) in str.lower(act['name'])) and  (act['activity type'] == activity_type)]
            else:
                activity_type = 'ordinary transforming activity'
                list_acts_in_eco = [act for act in eidb if (act['location'] == name_exc.split('{')[1].split('}')[0]) and (str.lower(name_exc.split('|')[0].split('{')[0][:-1]) in str.lower(act['name']))]

            if len(list_acts_in_eco) == 1:
                exc_to_update = list_acts_in_eco[0]
                updated_input = find_valid_inputs_ei(exc_to_update, row['amount'])

                for exc, adjusted_amount in updated_input:
                    list_data.append({  "activity": name, 'location': location, 'code' : code, "new_input": exc.input["name"],"amount": adjusted_amount, "unit": exc.input["unit"]   })
                dfa = pd.DataFrame(list_data)

                updated_activities.append(dfa)
                
            else:
                print('there were ', len(list_acts_in_eco), ' activities in ecoinvent that correspond to ', name_exc)
                print(list_acts_in_eco)


        else: # activities that need to go deeper in agribalyse itself
            name_exc = row['new_input']
            list_acts_in_agb = [a for a in dict_agb_miscunits_acts.keys() if a == name_exc]

            if (len(list_acts_in_agb) > 1):
                print(name_exc, ' got too many matches')
            if (len(list_acts_in_agb) == 0):
                print(name_exc, ' has no matches!')
            
            else:
                exc_to_update = list_acts_in_agb[0]
                updated_input = find_valid_inputs_agb(exc_to_update, row['amount'])
                if len(updated_input) > 0:
                    updated_activities.append(pd.concat(updated_input))

    df_return = pd.concat(updated_activities)
    df_return = df_return[~df_return['new_input'].str.contains('|'.join(['market for agricultural machinery', 'tractor', 'Trailer', 'Packaging', 'Emissions', 'metals uptake', 'seed', 'seedling', 'Harvester', 'General machinery', 'Jasmine rice', 'Spring ', 'Winter ', 'Chickpea', 'Triticale', 'Soft wheat', 'Oat grain', 'Onion sets', 'Durum wheat', 'Blue lupine', 'Spelt']))]

    return df_return


def apply_buckets(df):


    # Soil imporemevent
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['gypsum', 'lime', 'dolomite', 'limestone', 'vinasse', 'compost', 'Compost', 'filter cake', 'peat', 'calcium', 'lime', 
                                                                        'perlite', 'husk', 'ash','perlite', 'Sand', 'Polyethylene', 'Wheat straw', 'Organic amendment']), case=False)), ['bucket', 'subbucket']] = ['soil_improvement', 'soil_improvement']
    # Water inputs
    df.loc[(df['unit'] == 'm3') & (df['new_input'].str.contains('|'.join(['irrigating', 'Irrigation']), case=False)), ['bucket', 'subbucket']] = ['water_use', 'irrigation']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('Tap water', case=False)), ['bucket', 'subbucket']] = ['water_use', 'tap_water']
    df.loc[(df['unit'] == 'm3') & (df['new_input'].str.contains('Water', case=False)), ['bucket', 'subbucket']] = ['water_use', 'irrigation']
    # Plant protection
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['2,4-dichlorophenol', '2,4-dichlorotoluene', 'chlorine dioxide', 'trichloromethane','phenol', 'propylene glycol, liquid',
                                                                                'anthranilic', 'chloroform', '1,3-dichloropropene', 'alkylbenzene', 'borax', 'boric', 'alkylbenzene, linear','chemical, organic', 
                                                                                'borates', 'kaolin', 'naphtha', 'copper', 'tebuconazole', 'chlorothalonil', 'fosetyl-Al', 'nitrile-compound', 'urea-compound', 
                                                                                'phenoxy-compound', 'phthalimide-compound', 'pyridazine-compound', 'carbamate-compound', 'dithiocarbamate-compound', 'aclonifen', 
                                                                                'diphenylether-compound', 'pendimethalin', 'mancozeb', 'diazole-compound', 'dinitroaniline-compound', 'maleic hydrazide', 'triazine-compound', 
                                                                                'benzo[thia]diazole-compound', 'dimethenamide', 'captan', 'organophosphorus-compound, unspecified', 'atrazine', 'bipyridylium-compound', 
                                                                                'metolachlor', 'pyridine-compound', 'benzimidazole-compound', 'acetamide-anillide-compound, unspecified', 'benzoic-compound', 
                                                                                'pyrethroid-compound', 'glyphosate', 'diazine-compound', 'pesticide, unspecified', 'metaldehyde', 'cyclic', 'ethoxylated', 'Fungicide', 
                                                                                'Herbicide', 'Insecticide', 'Pesticide', 'Methylchloride', 'Benzoic acid', 'Pyrazole', 'Napropamide', 'Paraffin', 'Metamitron', 
                                                                                'pyrethroid-compound', 'Isoproturon', 'Folpet',  'chlorotoluron', 'Chlorotoluron', 'Horsetail decoction']), case = False)), ['bucket', 'subbucket']] = ['plant_protection', 'protection']

    # Fertiliser input
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['zinc', 'manganese', 'copper', 'sulfate' ,'boron', 'sulfite', 'fertiliser', 'sulfuric', 'magnesium', 'manganese','portafer', 
                                                                                'stone meal', 'Sulfur', 'market for sulfur', 'cobalt', 'magnesium', 'molybdenum', 'nickel', 'zinc']), case = False)), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & ((df['location'] == 'ZA')) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & ((df['location'].str.contains('BR-'))) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Manure', 'manure']))), ['bucket', 'subbucket']] = ['fertiliser_use', 'manure']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Fodder yeast', 'Carbon dioxide', 'Algae', 'Horn manure']), case=False)), ['bucket', 'subbucket']] = ['fertiliser_use', 'stimulant']

    # Planting and tillage activities
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('waste')), ['bucket', 'subbucket']] = ['planting & tillage activities', 'maintenance']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('seed')), ['bucket', 'subbucket']] = ['planting & tillage activities', 'seed']
    # Harvest activities
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['lubricating oil','vegetable oil', 'cart', 'grading']))), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    df.loc[(df['unit'] == 'l') & (df['new_input'].str.contains('drying')), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    # Misc
    df.loc[(df['unit'] == 'm3') & (df['new_input'].str.contains('|'.join(['fodder', 'bauxite', 'rape oil', 'alcohol', 'polyethylene']))), ['bucket', 'subbucket']] = ['misc', 'misc']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['polystyrene, expandable', 'polydimethylsiloxane', 'polyethylene, high density, granulate', 'stone wool', 'waste glass','packaging, for fertilisers', 'packaging film, low density polyethylene', 'packaging, for fertilisers or pesticides', 'packaging, for pesticides']))), ['bucket', 'subbucket']] = ['misc', 'inf/equipment']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['polystyrene, expandable', 'polydimethylsiloxane', 'polyethylene, high density, granulate', 'extrusion, plastic film', 'Steel', 'Polypropylene', 'polyurethane', 'Tractor', 'Agricultural machinery', 'Aluminium', 'market for tractor', 'market for agricultural machinery', 'Polystyrene', 'Refrigerant', 'Coconut', 'Polyvinylchloride', 'Silicone', 'Acetic acid', 'Lead', 'Nitric acid']), case=True)), ['bucket', 'subbucket']] = ['misc', 'inf/equipment']
    # Energy
    df.loc[(df['unit'] == 'kWh') & (df['new_input'].str.contains('|'.join(['electricity', 'Electricity']), case=False)), ['bucket', 'subbucket']] = ['energy', 'electricity']     
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('|'.join(['heat', 'Natural gas', 'burned in furnace']), case=False)), ['bucket', 'subbucket']] = ['energy', 'heat']     
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('|'.join(['diesel', 'petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('|'.join(['diesel-electric']), case=False)), ['bucket', 'subbucket']] = ['energy', 'electricity']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Diesel', 'petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Diesel', 'Kerosene', 'Petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    # Yield
    df.loc[(df['unit'] == 'kg/ha') & (df['new_input'].str.contains('yield', case=False)), ['bucket', 'subbucket']] = ['yield', 'yield']
    

def convert_units(df, subbucket, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria'''
    mask = (df['subbucket'] == subbucket) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef  # Update amount first
    df.loc[mask, 'unit'] = desired_unit  # Now update the unit
    return df


def convert_units_name_based(df, name, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria'''
    mask = (df['new_input'].astype(str).str.contains(name)) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef
    df.loc[mask, 'unit'] = desired_unit
    return df


# 1) Create new project, import agribalyse
bw.projects.set_current('name')
my_bio = bw.Database('ecoinvent biosphere - version')
eidb = bw.Database('ecoinvent database - version')
files_list = [file for file in glob.glob('PATH TO AGRIBALYSE')]
print('Section 1 Done')

# 2) Import agribalyse processes with units other than mass, energy, volume 
# because they are used as background for crop production activities
list_ag_miscunits = []

for file in glob.glob('PATH TO DATAFRAME OF ACTIVITIES WITH MISCALLENOUS UNITS'):
    list_ag_miscunits.append(pd.read_excel(file))

df_agb_miscunits = pd.concat(list_ag_miscunits)
df_agb_miscunits.columns = ['column1', 'column2', 'column3', 
                            'column4', 'column5', 'column6', 
                            'column7', 'column8', 'column9', 
                            'column10', 'column11']

df_agb_miscunits = df_agb_miscunits.reset_index().drop(columns = ['index'])
dict_agb_miscunits_acts = {}
process_start_index_miscunits = df_agb_miscunits.index[df_agb_miscunits['column1'] == 'Process'].tolist()

for i in tqdm(process_start_index_miscunits[:-1]):
    df_process = df_agb_miscunits.loc[i:process_start_index_miscunits[process_start_index_miscunits.index(i)+1]-1].copy()
    if find_value_for_keyword(df_process, 'Category type') != 'Waste treatment':
        name = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
        if type(name) != str:
            print(i, ' has a weird name')
        dict_agb_miscunits_acts[name] = i
print('Section 2 Done')


# 3) Filter crop production activities
list_dfs = []
for file in files_list:
    df_agb = pd.read_excel(file)
    df_agb.columns = ['column1', 'column2', 'column3', 'column4', 
                      'column5', 'column6', 'column7', 'column8', 'column9']
    list_dfs.append(df_agb)

df_agb = pd.concat(list_dfs).reset_index().drop(columns = 'index')
list_bioless, list_problematic, list_duplicates, list_sus_techno = ([] for i in range (4))
dict_acts = {}
dict_codes = {}
process_start_index = df_agb.index[df_agb['column1'] == 'Process'].tolist()
for i in tqdm(process_start_index[:-1]):
    data = []
    df_process = df_agb.loc[i:process_start_index[process_start_index.index(i)+1]-1].copy()
    code = find_value_for_keyword(df_process, 'Process identifier')
    name = find_value_for_keyword(df_process, 'Process name')
    name_product = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
    duplicate_check = find_value_for_keyword(df_process, 'Collection method')

    if ('ecoinvent' in str(duplicate_check).lower()) | ('adapted from wfldb' in str(duplicate_check).lower()) | ('ecoinvent sampling procedure' in str(duplicate_check).lower())| ('data based on nemecek and' in str(duplicate_check).lower()) :
        print(name_product, 'is a duplicate from other databases')
        list_duplicates.append(code)
    else:
        if all(x not in name_product for x in ['first production years', 'chicory witloof']) and\
            all(x not in str(duplicate_check).lower() for x in ['ecoinvent sampling procedure', 'data based on nemecek and', 'adapted from wfldb']):

            dict_codes[name_product] = code
            location = name_product.split('{')[1].split('}')[0]
            if type(name) != str: # sometimes the intended name field is empty
                name = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
            normalization_factor = find_normalization_value(df_process, name)
            df_bio = get_biosphere(df_process, normalization_factor)
            unit = find_product_unit(df_process, name)

            if (len(df_bio) > 0) & (unit in ['kg', 'ton']) & (all(word not in str.lower(name) for word in ['storage agency', 'intercrop', 'seedling', 'storehouse', ' seed', 'straw,'])):
                    df_techno = get_technosphere(df_process, normalization_factor)
                    if len(df_techno) < 5:
                        list_sus_techno.append(name)
                    df_techno_recursive = one_level_deeper_techno(df_techno, name, code)
                    if (df_bio['new_input'].astype(str).str.contains('Occupation').sum()):
                        data.append(pd.concat([df_bio.loc[df_bio['new_input'].str.contains('Water')] , pd.DataFrame(data = [{'new_input' : 'yield', 'amount' : ((10000) / (df_bio.loc[df_bio['new_input'].str.contains('Occupation'), 'amount'].values[0])), 'unit' : 'kg/ha' , 'code' : code, 'activity' : name, 'location' : location}])]))
                    else:
                        data.append(pd.concat([df_bio.loc[df_bio['new_input'].str.contains('Water')] , pd.DataFrame(data = [{'new_input' : 'yield', 'amount' : normalization_factor, 'unit' : 'kg/ha' , 'code' : code, 'activity' : name, 'location' : location}])]))
                    data.append(df_techno_recursive)
                    dict_acts[name] = data
            else:
                list_bioless.append(name)

acts_seed = [a for a in dict_acts.keys() if (' seed' in a) or ('seeds' in a) or ('seedling' in a) or ('seedlings' in a)]
print('Shit, we still need to take care of ', acts_seed)

with open('list_agb.pkl', 'wb') as g:
    pickle.dump(list(dict_acts.keys()), g)
print('Section 3 Done')

# 4) Convert units and harmonize their spelling
all_dfs = [df for df_list in dict_acts.values() for df in df_list]
df_final = pd.concat(all_dfs, ignore_index=True)

df_final.loc[df_final['unit'] == 'kilogram', 'unit'] = 'kg'
df_final.loc[df_final['unit'] == 'ton', 'amount'] = df_final.loc[df_final['unit'] == 'ton', 'amount']*1000
df_final.loc[df_final['unit'] == 'ton', 'unit'] = 'kg'
df_final.loc[df_final['unit'] == 'l', 'amount'] = df_final.loc[df_final['unit'] == 'l', 'amount']/1000
df_final.loc[df_final['unit'] == 'l', 'unit'] = 'm3'

df_summed_agb = df_final.groupby(['activity','new_input', 'unit', 'location', 'code'], as_index = False).sum().copy()
df_summed_filtered_agb = df_summed_agb.loc[~df_summed_agb['new_input'].str.contains('|'.join(['tractor', 'Tractor', 'trailer', 'tenure', 'harvester', 
                                                                                              'wood preservation', "harvesting cart", 'potato grading',
                                                                                                'cast iron', 'cleft timber', 'for agricultural machinery', 
                                                                                                'Soybean', 'Vanilla', 'Garlic', 'Ginger', 'Biomass burning', 
                                                                                                'Dehydration', 'combustion', 'Starch potato']))]
df_summed_filtered_agb = df_summed_filtered_agb.loc[df_summed_filtered_agb['amount'] > 0]
print('Section 4 Done')

# 5) Convert fertiliser input into NPK (active ingredient)
dict_npk = {'calcium ammonium nitrate (CAN) (with 27% N)':(0.27, 0, 0) ,
'calcium ammonium nitrate production':(0.27, 0, 0) ,
'market for calcium ammonium nitrate':(0.27, 0, 0) ,
'market for calcium nitrate':(0.171, 0, 0) ,
'calcium nitrate production':(0.171, 0, 0) ,
'market for urea ammonium nitrate mix' :( 0.301, 0, 0), 
'urea ammonium nitrate production' : (0.301, 0, 0) ,
'Urea (with 46% N)' : (0.46, 0, 0),
'market for urea': (0.466, 0, 0), 
'market for ammonium sulfate': (0.212, 0, 0), 
'ammonium sulfate production': (0.212, 0, 0),
'Average mineral fertilizer, as N' : (1, 0, 0), 
'market for ammonium nitrate':(0.35, 0, 0),
'market for ammonia, anhydrous, liquid' : (0.822, 0, 0),
'market for ammonium nitrate phosphate':(0.22, 0.22, 0),
'market for NPK (15-15-15) fertiliser' : (0.15, 0.344, 0.181),
'market for phosphate rock, beneficiated': (0, 0.32, 0),
'market for monoammonium phosphate' : (0.084, 0.52, 0),
'market for single superphosphate' : (0, 0.21, 0),
'market for triple superphosphate' : (0, 0.46, 0),
'market for diammonium phosphate' : (0.18, 0.46, 0),
'nutrient supply from thomas meal' : (0, 1, 0),
'market for potassium sulfate' : (0, 0, 0.541), 
'market for potassium chloride' : (0, 0, 0.6),
'market for potassium nitrate' : (0.139, 0, 0.466),
'Potassium nitrate' : (0.139, 0, 0.466),
'Ammonia (with 100% NH3)' : (0.822, 0, 0),
'Ammonia (with 100% NH3), production mix': (0.822, 0, 0),
'Rendered animal by-products' : (0.098, 0.037, 0.008),
'Horn meal' : (0.12, 0.02, 0),
'Sludge, thickened' : (0.0038, 0.0027, 0.0005),
'as K2O' : (0, 0, 1),
'as N' : (1, 0, 0),
'as P2O5' : (0,1,0)}

rows_final = []
for name_npk in dict_npk.keys():
    rows_divided = []
    for index, row in (df_summed_filtered_agb[df_summed_filtered_agb['new_input'].str.contains(name_npk, regex = False)]).iterrows():
        rows = [row.copy(), row.copy(), row.copy()]
        rows[0]['amount'] = row['amount']*dict_npk[name_npk][0]
        rows[0] = pd.concat([rows[0], pd.Series({'bucket' : 'fertiliser_use', 
                                                 'subbucket' : 'N_fertiliser'})])

        rows[1]['amount'] = row['amount']*dict_npk[name_npk][1]
        rows[1] = pd.concat([rows[1], pd.Series({'bucket' : 'fertiliser_use', 
                                                 'subbucket' : 'P_fertiliser'})])

        rows[2]['amount'] = row['amount']*dict_npk[name_npk][2]
        rows[2] = pd.concat([rows[2], pd.Series({'bucket' : 'fertiliser_use', 
                                                 'subbucket' : 'K_fertiliser'})])

        rows_divided.extend(rows)
    rows_final.extend(rows_divided)
df_fert = pd.DataFrame(rows_final)
df_agb_fert = df_fert.groupby(['activity', 'code', 'location', 'unit', 'bucket', 'subbucket'], as_index = False).sum()
df_non_fert = df_summed_filtered_agb[~df_summed_filtered_agb['new_input'].str.contains('|'.join(dict_npk.keys()))]
df_non_fert = df_non_fert[~df_non_fert['new_input'].str.contains('Ammonia (with 100% NH3)', regex = False)]

apply_buckets(df_non_fert)
df_non_fert = df_non_fert.loc[~(df_non_fert['subbucket'].isin(['seed', 'seedling', 
                                                               'inf/equipment', 'misc']))]
df_ag = pd.concat([df_non_fert, df_agb_fert], axis = 0).sort_values(by = 'activity')
df_ag = df_ag[~df_ag['new_input'].str.contains('construction')]
df_ag_bucketized = df_ag.groupby([ 'code', 'activity', 'bucket','subbucket', 
                                  'unit', 'new_input', 'location']).agg('sum').reset_index()
df_ag_bucketized['crop'] = df_ag_bucketized['activity'].str.split(',').str[0]
print('Section 5 Done')

# 6) Convert units and harmonize their spellings
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'Diesel', 'kg','MJ', 42.8)
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'market for diesel', 'kg','MJ', 42.8) 
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'market group for diesel', 'kg','MJ', 42.8) 
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'Petrol, two-stroke blend', 'kg', 'MJ', 43.2)
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'Petrol, low-sulfur', 'kg', 'MJ', 42.5)
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'Petrol combustion, unleaded, in motor mower', 'kg', 'MJ', 42.5)
df_ag_bucketized = convert_units_name_based(df_ag_bucketized, 'Petrol, unleaded', 'kg', 'MJ', 42.5)
df_wf = convert_units(df_ag_bucketized, 'tap_water', 'kg', 'm3', 0.001)

df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Autumn irrigated leek', 'crop'] = 'Leek'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Annual vining pea for industry', 'crop'] = 'Pea'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Baled hay', 'crop'] = 'Hay'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Barley straw', 'crop'] = 'Barley straw'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Beetroot for juice', 'crop'] = 'Beetroot'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Feed barley', 'crop'] = 'Barley'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Grain maize', 'crop'] = 'Maize'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Maize grain', 'crop'] = 'Maize'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Kiwifruit FR', 'crop'] = 'Kiwi'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'French bean', 'crop'] = 'Bean'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Silage maize', 'crop'] = 'silage Maize'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Wine grape system number 5 (phase)', 'crop'] = 'Wine grape'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Wine grape system number 4 (phase)', 'crop'] = 'Wine grape'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Wine grape system number 3 (phase)', 'crop'] = 'Wine grape'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Wine grape system number 2 (phase)', 'crop'] = 'Wine grape'
df_ag_bucketized.loc[df_ag_bucketized['crop'] == 'Wine grape system number 1 (phase)', 'crop'] = 'Wine grape'
print('Section 6 Done')

# 7) Construct inputs dataframe
df_final_agb = df_ag_bucketized.pivot_table(
    index='code', 
    columns=['subbucket', 'unit'], 
    values='amount', 
    aggfunc='sum')

df_final_agb.columns = [f"{subbucket}_{unit}" for subbucket, unit in df_final_agb.columns]
df_final_agb = df_final_agb.drop(columns = ['maintenance_kg', 'harvest_activities_kg'])

with open('df_agb_inputs.pkl', 'wb') as f:
    pickle.dump(df_final_agb, f)
print('Section 7 Done')

# 8) Construct categorical data dataframe
df_categorical_ag = df_ag_bucketized.reset_index()[['code', 'crop', 'location']].drop_duplicates(subset = ['code'])
df_categorical_ag = df_categorical_ag.rename(columns= {'location' : 'country'})
with open('df_agb_categorical.pkl', 'wb') as f:
    pickle.dump(df_categorical_ag, f)
with open('df_agb_categorical_complete.pkl', 'wb') as f:
    pickle.dump(df_ag_bucketized, f)
print('Section 8 Done')

# 9) Construct impacts dataframe for all three perspectives
# Hierarchist
files_list = glob.glob('path to exported impacts of crop production - H perspective')
excels_list = [pd.read_excel(file) for file in files_list]

list_impacts_midpoint = []
list_impacts_endpoint = []

for file in files_list:
    if 'mid' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Water consumption', 'Global warming']].set_index('impact')
        list_impacts_midpoint.append(a)

    if 'end' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]   
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Ecosystems']].set_index('impact')
        list_impacts_endpoint.append(a)

df_impacts_mid_ag = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_ag = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_ag = pd.concat([df_impacts_mid_ag, df_impacts_end_ag], axis = 1)
list_codes = []

for name in df_impacts_ag.index:
    code = dict_codes.get(name, np.nan)
    list_codes.append(code)

df_impacts_ag.index = list_codes
df_impacts_ag = df_impacts_ag[~df_impacts_ag.index.isna()]
df_impacts_agb = pd.merge(df_impacts_ag , df_ag_bucketized.drop_duplicates('activity') ,
                           left_index = True , right_on = 'code')[['Water consumption', 
                                                                   'Global warming', 
                                                                   'Ecosystems', 'code']].set_index('code')
with open('df_agb_impacts_H.pkl', 'wb') as f:
    pickle.dump(df_impacts_agb, f)
print('Section 9 Done')

# Egaletarian
files_list = glob.glob('path to exported impacts of crop production - E perspective')
excels_list = [pd.read_excel(file) for file in files_list]


list_impacts_midpoint = []
list_impacts_endpoint = []

for file in files_list:
    if 'mid' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Water consumption', 'Global warming']].set_index('impact')
        list_impacts_midpoint.append(a)

    if 'end' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]      
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Ecosystems']].set_index('impact')
        list_impacts_endpoint.append(a)

df_impacts_mid_ag = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_ag = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_ag = pd.concat([df_impacts_mid_ag, df_impacts_end_ag], axis = 1)
list_codes = []

for name in df_impacts_ag.index:
    code = dict_codes.get(name, np.nan)
    list_codes.append(code)

df_impacts_ag.index = list_codes
df_impacts_ag = df_impacts_ag[~df_impacts_ag.index.isna()]

df_impacts_agb = pd.merge(df_impacts_ag , df_ag_bucketized.drop_duplicates('activity') ,
                          left_index = True , right_on = 'code')[['Water consumption', 'Global warming', 
                                                                  'Ecosystems', 'code']].set_index('code')

with open('df_agb_impacts_E.pkl', 'wb') as f:
    pickle.dump(df_impacts_agb, f)
print('Section 10 Done')

# Individualist
files_list = glob.glob('path to exported impacts of crop production - I perspective')
excels_list = [pd.read_excel(file) for file in files_list]

list_impacts_midpoint = []
list_impacts_endpoint = []

for file in files_list:
    if 'mid' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Water consumption', 'Global warming']].set_index('impact')
        list_impacts_midpoint.append(a)

    if 'end' in file:
        a = pd.read_excel(file).dropna(how = 'all', axis = 1).T
        a = a.set_axis(a.iloc[0], axis=1)[2:]
        a = a.rename(columns = {'Damage category' : 'impact', 'Impact category' : 'impact'})
        a = a[['impact', 'Ecosystems']].set_index('impact')
        list_impacts_endpoint.append(a)

df_impacts_mid_ag = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_ag = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_ag = pd.concat([df_impacts_mid_ag, df_impacts_end_ag], axis = 1)
list_codes = []

for name in df_impacts_ag.index:
    code = dict_codes.get(name, np.nan)
    list_codes.append(code)

df_impacts_ag.index = list_codes
df_impacts_ag = df_impacts_ag[~df_impacts_ag.index.isna()]
df_impacts_agb = pd.merge(df_impacts_ag , df_ag_bucketized.drop_duplicates('activity') , 
                          left_index = True , right_on = 'code')[['Water consumption', 'Global warming',
                                                                   'Ecosystems', 'code']].set_index('code')

with open('df_agb_impacts_I.pkl', 'wb') as f:
    pickle.dump(df_impacts_agb, f)

print('Script is Done')