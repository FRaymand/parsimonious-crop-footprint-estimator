'''
This script:
1) Creates a new brightway project, imports WFLDB
2) Filters WFLDB activities to get crop production LCIs
3) Extracts technosphere/biosphere flows of crop production activities
4) Converts units and harmonizes unit spellings
5) Converts fertiliser inputs to active ingredients (NPK) & aggregates predictors
6) Saves df_wfldb_inputs
7) Constructs df_wfldb_categorical
8) Constructs df_wfldb_impacts for all three perspectives

N.B. For WFLDB, yield is just the normalization factor
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import brightway2 as bw
import sympy
import glob

def find_value_for_keyword(df, keyword):
    if df.loc[df['column1'] == keyword , 'column2'].values[0]:
        value = df.loc[df['column1'] == keyword , 'column2'].values[0]
    return value


def find_products(df):
    ''' used to find the name of product(s) and (their) units'''
    i, j = (df[df['column1'].isin(['Products','Avoided products'])].index)
    df = df.loc[i+1:j-1, ['column1', 'column3']].dropna(axis = 0, how = 'all')
    return {'name' : df['column1'].values, 'unit' : df['column3'].values}    


def find_normalization_value(df, name):
    return df.loc[df.index[df['column1'] == 'Products'] + 1, 'column2'].iloc[0]


def find_product_unit(df, name):
    return df.loc[df.index[df['column1'] == 'Products'] + 1, 'column3'].iloc[0]


def get_biosphere(df, normalization_factor, code, name, location):
    i, j = (df[df['column1'].isin(['Resources','Materials/fuels'])].index)
    df_bio = df.loc[i+1:j-1, ['column1', 'column2', 
                              'column3', 'column4', 'column5',
                              'column6','column7' ]].dropna(axis = 0, how = 'all')
    df_bio.columns = ['new_input','medium',  'amount', 'unit',
                       'distribution', 'sd', 'description']
    df_bio['amount'] = [float(sympy.sympify(field)) if type(field) != float else field for field in df_bio['amount']]
    df_bio['amount'] = df_bio['amount'] / normalization_factor
    df_bio['code'] = code
    df_bio['activity'] = name
    df_bio['location'] = location
    df_bio = df_bio.drop(columns = ['medium', 'distribution', 'sd'])
    return df_bio


def get_technosphere(df, normalization_factor, code, name, location):
    i, j = (df[df['column1'].isin(['Materials/fuels', 'Emissions to air'])].index)
    df_techno = df.loc[i+1:j-1,:]
    df_techno.columns = ['new_input', 'amount', 
                         'unit', 'description', 'description2', 
                         'description3', 'description4']
    df_techno = df_techno[['new_input', 'amount', 'unit', 'description']]
    df_techno = df_techno.dropna(subset = ['unit']).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
    if len(df_techno) > 0:
        df_techno.loc[:, 'amount'] = df_techno.loc[:, 'amount'] / normalization_factor
        df_techno.loc[:, 'code'] = code
        df_techno.loc[:, 'activity'] = name
        df_techno.loc[:, 'location'] = location 
    return df_techno


def get_biosphere(df, normalization_factor, code, name, location):
    i, j = (df[df['column1'].isin(['Resources','Materials/fuels'])].index)
    df_bio = df.loc[i+1:j-1, ['column1', 'column2', 'column3', 'column4', 'column5','column6','column7' ]].dropna(axis = 0, how = 'all')
    df_bio.columns = ['new_input','medium',  'amount', 'unit', 'description', 'description2', 'description3']
    df_bio['amount'] = [float(sympy.sympify(field)) if type(field) != float else field for field in df_bio['amount']]
    df_bio['amount'] = df_bio['amount'] / normalization_factor
    df_bio['code'] = code
    df_bio['activity'] = name
    df_bio['location'] = location
    df_bio = df_bio.drop(columns = ['medium', 'description2', 'description3'])
    return df_bio


def get_technosphere(df, normalization_factor, code, name, location):
    i, j = (df[df['column1'].isin(['Materials/fuels', 'Emissions to air'])].index)
    df_techno = df.loc[i+1:j-1,:]
    df_techno.columns = ['new_input', 'amount', 'unit', 'description', 'description2', 'description3', 'description4']
    df_techno = df_techno.dropna(subset = ['unit']).dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all')
    if len(df_techno) > 0:
        df_techno.loc[:, 'amount'] = df_techno.loc[:, 'amount'] / normalization_factor
        df_techno.loc[:, 'code'] = code
        df_techno.loc[:, 'activity'] = name
        df_techno.loc[:, 'location'] = location 
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
            adjusted_amount = exc.amount * parent_amount  # Multiply by parent amount
            
            if ('packaging' in (input_activity['name']).lower()) | (input_unit in {'m2', 'm2a'}):
                continue  # Skip this iteration if 'packaging' is found

            if (input_unit in ['cubic meter', 'm3']) & ('liquid manure spreading' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif (input_unit in {'litre', 'l'}) & ('drying' in input_activity['name']): # i want the electricity and/or heat inputs not the flow of drying (which is the removal of water)
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif input_unit in {'kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit', 'litre'}:
                valid_inputs.append((exc, adjusted_amount))

            elif (is_market_mix(input_activity)) & (input_unit not in {'kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit', 'litre'}):  
                # Dive deeper if it's a market mix AND not kg/MJ
                valid_inputs.extend(find_valid_inputs_ei(input_activity,adjusted_amount,  visited))
            elif (input_unit in {'hectare', 'Ha', 'ha'}) & ('trellis system' not in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))
            elif input_unit == 'hour':
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))
            elif (input_unit in {'ton kilometer', 'tkm'}) & ('transport, tractor and trailer, agricultural' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs_ei(input_activity, adjusted_amount, visited))

    return valid_inputs


def find_valid_inputs_wfldb(activity,activity_parent, parent_amount = 1, visited=None):
    """
    Recursively finds technosphere inputs that are in kg or MJ.
    If an input has another unit and is a market mix, it goes one level deeper.
    """
    if visited is None:
        visited = set()
    if activity in visited:  # Prevent infinite loops
        return []
    visited.add(activity)
    
    list_nonmatch = []
    valid_inputs = []
    df_process = list_processes[list_names.index(activity)].copy()
    name = activity
    location = name.split('/')[1].split(' ')[0]
 
    code = find_value_for_keyword(df_process, 'Process identifier')
    normalization_factor = find_normalization_value(df_process, name)
    unit = find_product_unit(df_process, name)
    if len(df_process) > 0:
        df_techno = get_technosphere(df_process, normalization_factor, code, name, location)
        if len (df_techno) == 0:
            print('uh oh!', name, ' has no technosphere? from activity')
        else:
            df_techno = df_techno[~df_techno['unit'].isin(['m2', 'm2a', np.nan])]

            for index, row in df_techno.iterrows():
                input_activity = row['new_input']
                adjusted_amount = row['amount'] * parent_amount
                input_unit = row['unit']
                if ('| Cut-off' in row['new_input']):
                    
                    input_activity_ei = [a for a in eidb if (input_activity.split('|')[1].split('|')[0][1:-1] == a['name']) & ( input_activity.split('{')[1].split('}')[0] in a['location'])][0]

                    if ('packaging' in (input_activity_ei['name']).lower()) | (input_unit in {'m2', 'm2a'}):
                        continue  # Skip this iteration if 'packaging' is found

                    if (input_unit in ['cubic meter', 'm3']) & ('liquid manure spreading' in input_activity_ei['name']):
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'],adjusted_amount,  visited))                  
                    elif (input_unit in {'litre', 'l'}) & ('drying' in (input_activity_ei['name']).lower()):
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'],adjusted_amount,  visited))
                    elif input_unit in {'kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit', 'litre'}:
                        valid_inputs.append(pd.DataFrame([{
                            "activity": activity_parent,
                            "location": location_act,
                            "code": code_og,
                            "new_input": input_activity_ei['name'],
                            "amount": adjusted_amount,
                            "unit": row["unit"]
                        }]))

                    elif (is_market_mix(input_activity_ei)) & (input_unit not in {'kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre", 'kg', 'MJ', 'm3', 'kWh', 'kwh', 'l', 'lit', 'litre'}):  
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'],adjusted_amount,  visited))
                    elif (input_unit in {'hectare', 'Ha', 'ha'}) & ('trellis system' not in input_activity_ei['name']):
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'], adjusted_amount, visited))
                    elif input_unit == 'hour':
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'], adjusted_amount, visited))
                    elif (input_unit in {'ton kilometer', 'tkm'}) & (input_activity_ei['name'] == 'transport, tractor and trailer, agricultural'):
                        valid_inputs.extend(find_valid_inputs_ei(input_activity_ei['name'], adjusted_amount, visited))

                else:
                    list_data_WFLDB = []
                    if (row['unit'] in ['cubic meter', 'm3']) & ('Irrigating,' in row['new_input']):
                        valid_inputs.extend(find_valid_inputs_wfldb(input_activity,activity_parent, adjusted_amount))  
                    elif (row['unit'] == 'ha') & ('trellis system' not in row['new_input']):
                        valid_inputs.extend(find_valid_inputs_wfldb(input_activity, activity_parent, adjusted_amount))
                    elif (row['unit'] in {'litre', 'l'}) & ('drying' in row['new_input'].lower()):
                        valid_inputs.extend(find_valid_inputs_wfldb(input_activity, activity_parent, adjusted_amount))
                    elif (row['unit'] in ['kg', 'ton', 'kWh', 'l', 'MJ', 'kilogram', 'litre', 'megajoule']):
                        list_data_WFLDB.append({"activity": activity_parent,'location': location_act,'code' : code_og, "new_input":input_activity, "amount": adjusted_amount, "unit": row["unit"] })
                        valid_inputs.append(pd.DataFrame(list_data_WFLDB))
                    else:
                        print(row['new_input'], 'has been omitted')              

    return valid_inputs


def one_level_deeper_techno(df, name, code_og):
    list_nonmatches = []
    updated_activities = []
    updated_activities.append(df[(df['unit'].isin(['kg', 'ton', 'kWh', 'MJ', 'kilogram','megajoule', 'g', 'gram'])) & (~df['new_input'].str.contains('solid manure loading'))])
    updated_activities.append(df[df['unit'].isin(['l', 'litre']) & (~df['new_input'].str.contains('Drying|drying'))])

    df = df[~(df['unit'].isin(['kg', 'ton', 'kWh', 'MJ', 'kilogram', 'megajoule', 'g', 'gram', 'm2', 'm2a', np.nan]) & (~df['new_input'].str.contains('solid manure loading')))] 
    df = df[~(df['new_input'].str.contains('|'.join(['seed', 'Seed', 'Seedling', 'seedling', 'Statistical', 'infrastructure', 'Emissions', 'Packaging', 'metals uptake'])))]
    df = df[~((df['unit'].isin(['l', 'litre'])) & (~df['new_input'].str.contains('Drying|drying')))]

    for index, row in df.iterrows():
        data = []
        name_exc = row['new_input']
        if ('| Cut-off' in  name_exc): # Taken from Ecoinvent
            list_acts_matching = [a for a in eidb if (name_exc.split('|')[1].split('|')[0][1:-1] == a['name']) & ( name_exc.split('{')[1].split('}')[0] in a['location'])]
            if len(list_acts_matching) > 1:
                print('too many matches for ', name_exc)
            elif len(list_acts_matching) == 0:
                list_nonmatches.append(name_exc)
                print('no matches for', name_exc)
                continue
            exc_matched = list_acts_matching[0]

            new_inputs = find_valid_inputs_ei(exc_matched, row['amount'])
            if len(new_inputs) > 0:

                for exc, adjusted_amount in new_inputs:
                    data.append({
                        "activity": name,
                        'location': location_act,
                        'code' : code_og,
                        'flow_code' : exc_matched['code'],
                        "new_input": exc.input["name"],
                        "amount": adjusted_amount,
                        "unit": exc.input["unit"]
                    })
                updated_input = pd.DataFrame(data)
                updated_activities.append(updated_input)

        else: # within WFLDB

            list_acts_matching = [a for a in list_names if a == name_exc]
            if (len(list_acts_matching) > 1):
                print(name_exc, ' got too many matches')
            if (len(list_acts_matching) == 0):
                print(name_exc, ' has no matches!')
            else:
                exc_to_update = list_acts_matching[0]
                updated_input = find_valid_inputs_wfldb(exc_to_update, name, row['amount'])
                if len(updated_input) > 0:
                    updated_activities.append(pd.concat(updated_input))

    df_return = pd.concat(updated_activities)
    df_return = df_return[~df_return['new_input'].str.contains('|'.join(['market for agricultural machinery', 'tractor',
                                                                         'trailer', 'Packaging', 'Emissions', 'metals uptake', 
                                                                         'seed', 'seedling', 'harvester', 'Cast iron']))]
    return df_return

def apply_buckets(df):
    # Energy
    df.loc[(df['unit'] == 'kWh') & (df['new_input'].str.contains('|'.join(['electricity', 'Electricity']), case=False)), ['bucket', 'subbucket']] = ['energy', 'electricity']
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('heat', case=False)), ['bucket', 'subbucket']] = ['energy', 'heat']
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('|'.join(['diesel', 'petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    df.loc[(df['unit'] == 'MJ') & (df['new_input'].str.contains('|'.join(['diesel-electric']), case=False)), ['bucket', 'subbucket']] = ['energy', 'electricity']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Diesel', 'petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Diesel', 'Kerosene', 'Petrol']), case=False)), ['bucket', 'subbucket']] = ['energy', 'fuel']
    # Soil imporemevent
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['gypsum', 'lime', 'dolomite', 'limestone', 'vinasse', 'compost', 'Compost', 'biowaste','filter cake', 'peat', 'calcium', 'lime', 
                                                                        'perlite', 'husk', 'ash','perlite', 'Sand', 'Polyethylene']), case=False)), ['bucket', 'subbucket']] = ['soil_improvement', 'soil_improvement']
    # Water inputs
    df.loc[(df['unit'] == 'm3') & (df['new_input'].str.contains('irrigating', case=False)), ['bucket', 'subbucket']] = ['water_use', 'irrigation']
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
                                                                                'Herbicide', 'Insecticide', 'Pesticide', 'Methylchloride', 'Benzoic acid', 'Pyrazole', 'Napropamide', 'Paraffin', 'Metamitron']))), ['bucket', 'subbucket']] = ['plant_protection', 'protection']

    # Fertiliser input
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['zinc', 'manganese', 'copper', 'sulfate' ,'boron', 'sulfite', 'fertiliser', 'sulfuric', 'magnesium', 'manganese','portafer', 
                                                                                'stone meal', 'Sulfur', 'market for sulfur', 'cobalt', 'magnesium', 'molybdenum', 'nickel', 'zinc']))), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & ((df['location'] == 'ZA')) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & ((df['location'].str.contains('BR-'))) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Manure', 'manure']))), ['bucket', 'subbucket']] = ['fertiliser_use', 'manure']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['Fodder yeast']), case=False)), ['bucket', 'subbucket']] = ['fertiliser_use', 'stimulant']
    # Planting and tillage activities
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('waste')), ['bucket', 'subbucket']] = ['planting & tillage activities', 'maintenance']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('seed')), ['bucket', 'subbucket']] = ['planting & tillage activities', 'seed']
    # Harvest activities
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['lubricating oil','vegetable oil', 'cart', 'grading']))), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    df.loc[(df['unit'] == 'l') & (df['new_input'].str.contains('drying')), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    # Misc
    df.loc[(df['unit'] == 'm3') & (df['new_input'].str.contains('|'.join(['fodder', 'bauxite', 'rape oil', 'alcohol', 'polyethylene']))), ['bucket', 'subbucket']] = ['misc', 'misc']
    df.loc[(df['unit'] == 'kg') & (df['new_input'].str.contains('|'.join(['polystyrene, expandable', 'polydimethylsiloxane', 'polyethylene, high density, granulate', 
                                                                          'stone wool', 'waste glass','packaging, for fertilisers', 'packaging film, low density polyethylene', 
                                                                          'packaging, for fertilisers or pesticides', 'packaging, for pesticides']))), ['bucket', 'subbucket']] = ['misc', 'inf/equipment']
    df.loc[(df['unit'] == 'kg/ha') & (df['new_input'].str.contains('yield', case=False)), ['bucket', 'subbucket']] = ['yield', 'yield']


def convert_units(df, subbucket, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria
    Converts the amounts first before modifying the unit
    '''
    mask = (df['subbucket'] == subbucket) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef
    df.loc[mask, 'unit'] = desired_unit
    return df

def convert_units_name_based(df, name, prev_subbucket, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria'''

    mask = (df['new_input'].str.contains(name)) & (df['unit'] == prev_unit) & (df['subbucket'] == prev_subbucket)
    df.loc[mask, 'amount'] *= coef  # Update amount first
    df.loc[mask, 'unit'] = desired_unit  # Now update the unit
    return df

# 1) We create a new brightway project and import WFLDB
bw.projects.set_current('name')
my_bio = bw.Database('ecoinvent biosphere - version')
eidb = bw.Database('ecoinvent database - version')

df_wfldb = pd.read_excel('Path To WFLDB')
df_wfldb.columns = ['column1', 'column2', 'column3', 
                    'column4', 'column5', 'column6', 'column7']
df_wfldb = df_wfldb.reset_index().drop(columns = ['index'])
print('Section 1 Done')

# 2) We filter WFLDB to get crop production activities
process_start_index = df_wfldb.index[df_wfldb['column1'] == 'Process'].tolist()
list_agri_processes , list_processes, list_names, list_names_agri = ([] for i in range(4))

for i in tqdm(process_start_index[:-1]):
    data = []
    df_process = df_wfldb.loc[i:process_start_index[process_start_index.index(i)+1]-1].copy()

    # For activities with biproducts, the first row represents the main product
    # such as wheat grain and wheat straw. Therefore by choosing only the
    # first row, we are automatically discarding wheat/oat/barley straws, 
    # coconut husks, and so on.
    name_product = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
    list_names.append(name_product)
    location = name_product.split('/')[1].split(' ')[0]
    list_processes.append(df_process)
    if ('at farm' in name_product)  and (location not in ['RoW', 'RER', 'RNA', 'GLO'])  and all(word not in name_product for word in ['Beef', 'feed',
                    'Silage' ,'bag','building','seedling',' seed', 'Lamb' , 'hen',
                    'cow', 'milk', 'Swine', 'Turkey', 'Broiler', 'Chicken', 'Hay',
                    'Honey', 'production mix', 'Production mix', 'Production Mix',
                    'sheep', 'Sheep', 'egg', 'Egg']):
        list_names_agri.append(name_product.split('/')[0])
        list_agri_processes.append(df_process)
print('Section 2 Done')

# 3) We extract biosphere and technosphere flows
dict_acts = {}

for df_process in tqdm(list_agri_processes): 
    data = []
    name = df_process.loc[df_process.loc[df_process['column1'] == 'Products'].index+1, 'column1'].values[0]
    location_act = name.split('/')[1].split(' ')[0]
    normalization_factor = find_normalization_value(df_process, name)
    code_og = find_value_for_keyword(df_process, 'Process identifier')
    df_bio = get_biosphere(df_process, normalization_factor, code_og, name, location_act)

    if len(df_bio) > 0:
        df_techno = get_technosphere(df_process, normalization_factor, code_og, name, location_act)
        df_techno = df_techno[~df_techno['new_input'].str.contains('|'.join(['Treillis system', 'Particleboard', 'EUR-flat', 'Statistical']))]
        if 'Coffee, green beans' in name:
            df_techno = df_techno[~df_techno['description'].str.contains('|'.join(['processing', 'drying']))]
            df_bio = df_bio[~df_bio['description'].str.contains('|'.join(['processing', 'drying']))]
            df_techno = df_techno[~df_techno['new_input'].str.contains('|'.join(['dehydration', 'Dehydration']))]
        df_techno_recursive = one_level_deeper_techno(df_techno, name, code_og)
        if len(df_techno_recursive) > 0:
            data.append(df_techno_recursive)
            data.append(pd.DataFrame(data = [{'new_input' : 'yield', 'amount' : normalization_factor, 'unit' : 'kg/ha' , 'code' : code_og, 'activity' : name, 'location' : location_act}]))
            data.append(df_bio.dropna(subset = 'amount')[df_bio.dropna(subset = 'amount')['new_input'].str.contains('Water')].groupby(['unit', 'code', 'activity', 'location']).sum().reset_index()[['new_input', 'amount', 'unit', 'code', 'activity', 'location']])
            dict_acts[name] = pd.concat(data)
    else:
        print(name, ' has no biosphere')

df = pd.concat(dict_acts).reset_index()[['new_input', 'amount','unit', 'code' ,'activity','location' ]]
list_wfldb = list(dict_acts.keys())
with open('list_wfldb.pkl', 'wb') as g:
    pickle.dump(list_wfldb, g)
print('Section 3 Done')

# 4) We convert units and harmonize unit spellings across databases
df.loc[df['unit'] == 'g', 'amount'] = df.loc[df['unit'] == 'g', 'amount']/1000
df.loc[df['unit'] == 'g', 'unit'] = 'kg'
df.loc[df['unit'] == 'ton', 'amount'] = df.loc[df['unit'] == 'ton', 'amount']*1000
df.loc[df['unit'] == 'ton', 'unit'] = 'kg'
df.loc[df['unit'] == 'kilogram', 'unit'] = 'kg'
df.loc[df['unit'] == 'kilowatt hour', 'unit'] = 'kWh'
df.loc[df['unit'] == 'megajoule', 'unit'] = 'MJ'
df.loc[df['unit'] == 'cubic meter', 'unit'] = 'm3'
df.loc[df['unit'] == 'l', 'amount'] = df.loc[df['unit'] == 'l', 'amount']/1000
df.loc[df['unit'] == 'l', 'unit'] = 'm3'

df_summed_wfldb = df.groupby(['activity','new_input', 
                              'unit', 'location', 'code'], as_index = False).sum().copy()
df_summed_filtered_wfldb = df_summed_wfldb.loc[~df_summed_wfldb['new_input'].str.contains('|'.join(['tractor', 'Tractor', 'trailer', 'tenure', 
                                                                                                    'harvester', 'wood preservation', "harvesting cart", 
                                                                                                    'potato grading', 'cast iron', 'cleft timber', 'for agricultural machinery', 
                                                                                                    'Soybean', 'Vanilla', 'Garlic', 'Ginger', 'Biomass burning', 'Dehydration', 
                                                                                                    'Biomass burning', 'Biowaste treatment']))]
df_summed_filtered_wfldb = df_summed_filtered_wfldb.loc[df_summed_filtered_wfldb['amount'] > 0]
print('Section 4 Done')

# 5) Convert fertiliser input to NPK (active ingredient)
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
'Ammonia (with 100% NH3)' : (0.822, 0, 0),
'Ammonia (with 100% NH3), production mix': (0.822, 0, 0),
'as K2O' : (0, 0, 1),
'as N' : (1, 0, 0),
'as P2O5' : (0,1,0)}

rows_final = []
for name_npk in dict_npk.keys():
    rows_divided = []
    for index, row in (df_summed_filtered_wfldb[df_summed_filtered_wfldb['new_input'].str.contains(name_npk, regex = False)]).iterrows():
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
df_wf_fert = df_fert.groupby(['activity', 'code', 
                              'location', 'unit', 'bucket', 'subbucket'], as_index = False).sum()
df_non_fert = df_summed_filtered_wfldb[~df_summed_filtered_wfldb['new_input'].str.contains('|'.join(dict_npk.keys()))]
df_non_fert = df_non_fert[~df_non_fert['new_input'].str.contains('Ammonia (with 100% NH3)', regex = False)]
apply_buckets(df_non_fert)
df_non_fert = df_non_fert.loc[~(df_non_fert['subbucket'].isin(['seed',
                                                                'seedling', 'inf/equipment', 'misc', 'maintenance']))]
print('Section 5 Done')

# 6) Save the inputs dataframe
df_wf = pd.concat([df_non_fert, df_wf_fert], axis = 0).sort_values(by = 'activity')

# The following conversion ratios are based on agribalyse/ecoinvent documentation
df_wf = convert_units_name_based(df_wf, 'diesel','fuel', 'kg','MJ', 42.8) 
df_wf = convert_units_name_based(df_wf, 'Petrol','fuel', 'kg', 'MJ', 43.2)
df_wf = convert_units_name_based(df_wf, 'petrol','fuel', 'kg', 'MJ', 43.2)
df_wf = convert_units_name_based(df_wf, 'acid methyl ester','fuel', 'kg','MJ', 36.8)
df_wf = convert_units_name_based(df_wf, 'Kerosene','fuel', 'kg', 'MJ', 43)
df_wf = convert_units_name_based(df_wf, 'peat moss','soil_improvement', 'm3', 'kg', 207)
df_wf = convert_units(df_wf, 'electricity', 'MJ', 'kWh', (1/3.6))
df_wf = convert_units(df_wf, 'tap_water', 'kg', 'm3', 0.001)

df_wf_full = df_wf.pivot_table(columns = [ 'subbucket', 'unit'], 
                               values = 'amount' , index = 'code', 
                               aggfunc = 'sum').sort_values(by = 'code').fillna (0)
df_wf_full.columns = [f"{subbucket}_{unit}" for subbucket, unit in df_wf_full.columns]

with open('df_wfldb_inputs.pkl', 'wb') as f:
    pickle.dump(df_wf_full, f)
print('Section 6 Done')


# 7) Make the categorical data dataframe
df_wf_bucketized = df_wf.groupby([ 'code', 'activity', 'bucket','subbucket', 'unit', 'location']).agg('sum').reset_index()
df_wf_bucketized['crop'] = df_wf_bucketized['activity'].str.split(', at').str[0]

df_wf_categorical = df_wf_bucketized.copy()
df_wf_categorical[['crop', 'etc']] = df_wf_categorical['activity'].str.split(', at', expand = True)
df_wf_categorical = df_wf_categorical.drop(columns = ['activity', 'etc'])
df_wf_categorical = df_wf_categorical.rename(columns = {'location' : 'country'})
df_wf_categorical = df_wf_categorical[['code', 'country', 'crop']]
df_wf_categorical = df_wf_categorical.drop_duplicates()

with open('df_wfldb_categorical.pkl', 'wb') as f:
    pickle.dump(df_wf_categorical, f)
with open('df_wfldb_categorical_complete.pkl', 'wb') as f:
    pickle.dump(df_wf_bucketized, f)
print('Section 7 Done')

# 8) We make the impact dataframes for all three perspectives
# 8.1) Egaletarian perspective
files_list = glob.glob('path to exported impacts of crop production - E perspective')
excels_list = [pd.read_excel(file) for file in files_list]
list_impacts_midpoint, list_impacts_endpoint = [], []

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

df_impacts_mid_wf = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_wf = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_end_wf = df_impacts_end_wf[df_impacts_end_wf.index.isin(df_wf['activity'])]
df_impacts_mid_wf = df_impacts_mid_wf[df_impacts_mid_wf.index.isin(df_wf['activity'])]

df_impacts_end_wf = df_impacts_end_wf.drop_duplicates()
df_impacts_mid_wf = df_impacts_mid_wf.drop_duplicates()

df_impacts_wf = pd.concat([df_impacts_mid_wf, df_impacts_end_wf], axis = 1)
df_impacts_wf = pd.merge(df_wf, df_impacts_wf, left_on = 'activity', right_index = True, how = 'inner').set_index('code')[['Water consumption', 'Global warming', 'Ecosystems']].drop_duplicates()
with open('df_wfldb_impacts_E.pkl', 'wb') as f:
    pickle.dump(df_impacts_wf, f)
print('Section 8.1 Done')


# 8.2) Individualist perspective
files_list = glob.glob('path to exported impacts of crop production - I perspective')
excels_list = [pd.read_excel(file) for file in files_list]
list_impacts_midpoint, list_impacts_endpoint = [], []

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

df_impacts_mid_wf = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_wf = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_end_wf = df_impacts_end_wf[df_impacts_end_wf.index.isin(df_wf['activity'])]
df_impacts_mid_wf = df_impacts_mid_wf[df_impacts_mid_wf.index.isin(df_wf['activity'])]

df_impacts_end_wf = df_impacts_end_wf.drop_duplicates()
df_impacts_mid_wf = df_impacts_mid_wf.drop_duplicates()

df_impacts_wf = pd.concat([df_impacts_mid_wf, df_impacts_end_wf], axis = 1)
df_impacts_wf = pd.merge(df_wf, df_impacts_wf, left_on = 'activity', right_index = True, how = 'inner').set_index('code')[['Water consumption', 'Global warming', 'Ecosystems']].drop_duplicates()

with open('df_wfldb_impacts_I.pkl', 'wb') as f:
    pickle.dump(df_impacts_wf, f)
print('Section 8.2 Done')


# 8.3) Hierarchist perspective
files_list = glob.glob('path to exported impacts of crop production - H perspective')
excels_list = [pd.read_excel(file) for file in files_list]
list_impacts_midpoint, list_impacts_endpoint = [], []

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

df_impacts_mid_wf = pd.concat(list_impacts_midpoint, axis = 0)
df_impacts_end_wf = pd.concat(list_impacts_endpoint, axis = 0)
df_impacts_end_wf = df_impacts_end_wf[df_impacts_end_wf.index.isin(df_wf['activity'])]
df_impacts_mid_wf = df_impacts_mid_wf[df_impacts_mid_wf.index.isin(df_wf['activity'])]

df_impacts_end_wf = df_impacts_end_wf.drop_duplicates()
df_impacts_mid_wf = df_impacts_mid_wf.drop_duplicates()

df_impacts_wf = pd.concat([df_impacts_mid_wf, df_impacts_end_wf], axis = 1)
df_impacts_wf = pd.merge(df_wf, df_impacts_wf, left_on = 'activity', right_index = True, how = 'inner').set_index('code')[['Water consumption', 'Global warming', 'Ecosystems']].drop_duplicates()

with open('df_wfldb_impacts_H.pkl', 'wb') as f:
    pickle.dump(df_impacts_wf, f)

print('Script Done')