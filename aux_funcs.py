import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import pickle
import copy

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

def convert_units(df, subbucket, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria'''
    mask = (df['subbucket'] == subbucket) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef  # Update amount first
    df.loc[mask, 'unit'] = desired_unit  # Now update the unit
    return df