'''
This script:
1) Creates a new brightway project, imports ecoinvent, imports ReCiPe methods
2) Filters ecoinvent to get crop production activities & makes a dataframe of them
3) Calculates footprints for H/E/I perspectives of ReCiPe
4) Compiles technosphere input flows 
5) Converts fertiliser inputs to active ingredients (NPK) & aggregates predictors
6) Applies conversions between units & harmonizes unit spelling
7) Extracts biosphere flows
8) Concatenates biosphere/technosphere flows into the inputs dataframe
9) Constructs the categorical dataframe for ecoinvent

N.B. Make sure you have the license for ecoinvent 3.11 to be able to reproduce 
these results (relevant to Section 1)
'''
from typing import Iterable, List
import pandas as pd
from tqdm import tqdm
import pickle
import re
import brightway2 as bw
import bw2data as bd
from ag_identification import get_land_flow_codes
from ag_identification import is_agriculture


def filter_landOccupation (list_activities: Iterable = None) -> List:
    '''Filters out activities that are not agricultural in terms of land 
    occupation
    '''
    assert (list_activities and isinstance(list_activities, Iterable), "Input should be an iterable of activities")

    return [act for act in tqdm(list_activities, mininterval = 2) if 
            is_agriculture(act , land_flow_codes = land_flow_codes_auto)]

def filter_animalHusbandry (list_activities):
    ''' Filters out activities related to animal husbandry
     all activities should be in a list and have a "name" key to be used with this function
    '''
    return [act for act in list_activities if all(animal not in act['name'] for animal in animal_list)]

def filter_irrelevant (list_activities):
    ''' Filters out irrelevant/ignorable activities.
    all activities should be in a list and have a "name" key to be used with this function 
    '''
    return [act for act in list_activities if all(re.search(r'\b'+re.escape(keyword)+r'\b' 
                        , act['name']) is None for keyword in irrelevant_list)]

def filter_names (list_activities):
    return [act for act in list_activities if 'production' in act['name'] 
                and any(commodity in act['name'] for commodity in commodities_agricultural) 
                and all(re.search((r'\b' + commo + r'\b'), act['name']) == None for commo in commodities_exception) ]

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

def find_valid_inputs(activity, parent_amount = 1, visited=None):
    """
    Recursively finds technosphere inputs that are in kg, MJ, kWh, m3 and lit.
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
            
            if 'packaging' in input_activity['name']:
                continue  # Skip this iteration if 'packaging' is found

            if "green manure" in input_activity["name"].lower():
                valid_inputs.append((exc, adjusted_amount * 2500  ))
                exc.input["unit"] = "kilogram"  # convert unit to kg
                continue  # Skip further checks for this input

            if (input_unit == 'cubic meter') & ('liquid manure spreading' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs(input_activity,adjusted_amount,  visited))

            elif (input_unit == 'litre') & ('drying' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs(input_activity,adjusted_amount,  visited))

            elif input_unit in ['kilogram', "megajoule", "cubic meter", "kilowatt hour", "litre"]:
                # if 'machinery' not in input_activity['name']:
                valid_inputs.append((exc, adjusted_amount))
            elif is_market_mix(input_activity):  
                # Dive deeper if it's a market mix AND not kg/MJ
                valid_inputs.extend(find_valid_inputs(input_activity,adjusted_amount,  visited))

            elif (input_unit == 'hectare') & ('trellis system' not in input_activity['name']) & ('land use change' not in input_activity['name'].lower()):
                valid_inputs.extend(find_valid_inputs(input_activity, adjusted_amount, visited))
            elif input_unit == 'hour':
                valid_inputs.extend(find_valid_inputs(input_activity, adjusted_amount, visited))
            elif (input_unit in {'ton kilometer', 'tkm'}) & ('transport, tractor and trailer, agricultural' in input_activity['name']):
                valid_inputs.extend(find_valid_inputs(input_activity, adjusted_amount, visited))

    return valid_inputs


def apply_buckets(df):

    # Energy
    df.loc[(df['unit'] == 'kilowatt hour') & (df['new_input'].str.contains('electricity')), ['bucket', 'subbucket']] = ['energy', 'electricity']
    df.loc[(df['unit'] == 'megajoule') & (df['new_input'].str.contains('heat')), ['bucket', 'subbucket']] = ['energy', 'heat']
    df.loc[(df['unit'] == 'megajoule') & (df['new_input'].str.contains('machinery')), ['bucket', 'subbucket']] = ['energy', 'fuel']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['petrol', 'kerosene', 'fatty', 'diesel']))), ['bucket', 'subbucket']] = ['energy', 'fuel']

    # Soil improvement
    df.loc[(df['unit'] == 'cubic meter') & (df['new_input'].str.contains('peat')), ['bucket', 'subbucket']] = ['soil_improvement', 'soil_improvement']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['gypsum', 'lime', 'dolomite', 'limestone', 'vinasse', 'compost', 
                                                                                'filter cake', 'peat', 'calcium', 'lime', 'perlite', 'husk', 'ash']))), ['bucket', 'subbucket']] = ['soil_improvement', 'soil_improvement']
    # Water inputs
    df.loc[(df['unit'] == 'cubic meter') & (df['new_input'].str.contains('irrigation')), ['bucket', 'subbucket']] = ['water_use', 'irrigation']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('tap water')), ['bucket', 'subbucket']] = ['water_use', 'tap_water']
    # Plant protection
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['2,4-dichlorophenol', '2,4-dichlorotoluene', 'chlorine dioxide', 'trichloromethane','phenol', 'propylene glycol, liquid',
                                                                                'anthranilic', 'chloroform', '1,3-dichloropropene', 'alkylbenzene', 'borax', 'boric', 'alkylbenzene, linear','chemical, organic', 
                                                                                'borates', 'kaolin', 'naphtha', 'copper', 'tebuconazole', 'chlorothalonil', 'fosetyl-Al', 'nitrile-compound', 'urea-compound', 
                                                                                'phenoxy-compound', 'phthalimide-compound', 'pyridazine-compound', 'carbamate-compound', 'dithiocarbamate-compound', 'aclonifen', 
                                                                                'diphenylether-compound', 'pendimethalin', 'mancozeb', 'diazole-compound', 'dinitroaniline-compound', 'maleic hydrazide', 'triazine-compound', 
                                                                                'benzo[thia]diazole-compound', 'dimethenamide', 'captan', 'organophosphorus-compound, unspecified', 'atrazine', 'bipyridylium-compound', 
                                                                                'metolachlor', 'pyridine-compound', 'benzimidazole-compound', 'acetamide-anillide-compound, unspecified', 'benzoic-compound', 
                                                                                'pyrethroid-compound', 'glyphosate', 'diazine-compound', 'pesticide, unspecified', 'metaldehyde', 'cyclic', 'ethoxylated',
                                                                                'folpet', 'napropamide', 'metamitron', 'prosulfocarb', 'isoproturon', 'chlorotoluron', 'potassium carbonate']))), ['bucket', 'subbucket']] = ['plant_protection', 'protection']
    # Fertiliser input
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['zinc', 'manganese', 'copper', 'sulfate' ,'boron', 'sulfite', 'fertiliser', 'sulfuric', 'magnesium', 'manganese','portafer', 
                                                                                'stone meal', 'Sulfur', 'market for sulfur', 'cobalt', 'magnesium', 'molybdenum', 'nickel', 'zinc']))), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['sodium silicate']), case=False)), ['bucket', 'subbucket']] = ['fertiliser_use', 'stimulant']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('manure')), ['bucket', 'subbucket']] = ['fertiliser_use', 'manure']
    df.loc[(df['unit'] == 'kilogram') & ((df['location'] == 'ZA')) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    df.loc[(df['unit'] == 'kilogram') & ((df['location'].str.contains('BR-'))) & (df['new_input'].str.contains('market for chemical, inorganic')), ['bucket', 'subbucket']] = ['fertiliser_use', 'micronutrients']
    # Planting and tillage activities
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('waste')), ['bucket', 'subbucket']] = ['planting & tillage activities', 'maintenance']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['seed', 'soybean']))), ['bucket', 'subbucket']] = ['planting & tillage activities', 'seed']
    # Harvest activities
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['lubricating oil','vegetable oil', 'cart', 'grading']))), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    df.loc[(df['unit'] == 'litre') & (df['new_input'].str.contains('drying')), ['bucket', 'subbucket']] = ['harvest_activities', 'harvest_activities']
    # Misc
    df.loc[(df['unit'] == 'cubic meter') & (df['new_input'].str.contains('|'.join(['fodder', 'bauxite', 'rape oil', 'alcohol', 'polyethylene']))), ['bucket', 'subbucket']] = ['misc', 'misc']
    df.loc[(df['unit'] == 'kilogram') & (df['new_input'].str.contains('|'.join(['polystyrene, expandable', 'polydimethylsiloxane', 'polyethylene, high density, granulate',
                                                                                 'stone wool', 'waste glass','packaging, for fertilisers', 'packaging film, low density polyethylene', 
                                                                                 'packaging, for fertilisers or pesticides', 'packaging, for pesticides']))), ['bucket', 'subbucket']] = ['misc', 'inf/equipment']


def convert_units(df, subbucket, prev_unit, desired_unit, coef):  
    '''Convert units based on flow names rather than subbucket or other criteria'''

    mask = (df['subbucket'] == subbucket) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef
    df.loc[mask, 'unit'] = desired_unit
    return df


def convert_units_name_based(df, name, prev_unit, desired_unit, coef):
    '''Convert units based on flow names rather than subbucket or other criteria'''

    mask = (df['new_input'].str.contains(name)) & (df['unit'] == prev_unit)
    df.loc[mask, 'amount'] *= coef  
    df.loc[mask, 'unit'] = desired_unit
    return df
        
# 1) We create a new brightway project and define the ReCiPe methods.
bd.projects.set_current('name')
my_bio = bw.Database('ecoinvent biosphere - version')
eidb = bw.Database('ecoinvent database - version')

recipe2016_PDF_tot_noLT = ('ReCiPe 2016 v1.03, endpoint (H) no LT', 
 'total: ecosystem quality no LT',  #10 multiple under-indentations
 'ecosystem quality no LT')

recipe2016_PDF_tot_noLT_E = ('ReCiPe 2016 v1.03, endpoint (E) no LT',
 'total: ecosystem quality no LT',
 'ecosystem quality no LT')

recipe2016_PDF_tot_noLT_I = ('ReCiPe 2016 v1.03, endpoint (I) no LT',
 'total: ecosystem quality no LT',
 'ecosystem quality no LT')

recipe2016_WU_noLT = ('ReCiPe 2016 v1.03, midpoint (H) no LT',
 'water use no LT',
 'water consumption potential (WCP) no LT')

recipe2016_WU_noLT_E = ('ReCiPe 2016 v1.03, midpoint (E) no LT',
 'water use no LT',
 'water consumption potential (WCP) no LT')

recipe2016_WU_noLT_I = ('ReCiPe 2016 v1.03, midpoint (I) no LT',
 'water use no LT',
 'water consumption potential (WCP) no LT')

recipe2016_CC_noLT_I = ('ReCiPe 2016 v1.03, midpoint (I) no LT',
 'climate change no LT',
 'global warming potential (GWP20) no LT')

recipe2016_CC_noLT = ('ReCiPe 2016 v1.03, midpoint (H) no LT',
 'climate change no LT',
 'global warming potential (GWP100) no LT')

recipe2016_CC_noLT_E = ('ReCiPe 2016 v1.03, midpoint (E) no LT',
 'climate change no LT',
 'global warming potential (GWP1000) no LT')

print('Section 1 completed')

# 2) Filter ecoinvent to get crop production activities & makes a dataframe of them
commodities_agricultural = ['abaca', 'alfalfa', 'almond', 'apple', 'apricot',
                            'areca', 'arracha', 'aroowroot', 'artichoke', 'asparagus', 
                            'avocado', 'bajra', 'banana', 'barley', 'beans', 
                            'bean', 'beet', 'bergamot', 'blackberry', 'blueberry', 
                            'breadfruit', 'broccoli', 'brussel sprout', 'buckwheat',
                            'cabbage', 'cantalope', 'caraway', 'cardamom', 'cardoon',
                            'carob', 'carrot', 'cashew', 'cassava', 'cauliflower', 
                            'celeriac', 'celery', 'chayote', 'cherry', 'chestnut', 
                            'chickpea', 'chicory', 'cinnamon', 'chilli', 'citron', 
                            'citronella', 'clementine', 'clove', 'clover', 'cocoa', 
                            'cacao', 'coconut', 'coffee', 'colza', 'coriander', 
                            'corn', 'cotton', 'cottonseed','cowpea', 'cranberry', 
                            'cress', 'cucumber', 'currant', 'custard apple',
                            'dasheen', 'dates', 'date', 'durra', 'sorghum', 
                            'durum wheat', 'edo', 'eggplant', 'courgette', 'endive', 
                            'fennel', 'fenugreek', 'fig', 'filbert', 'fique', 'hazelnut', 
                            'flax', 'formio', 'garlic', 'geranium', 'ginger', 
                            'gooseberry', 'gourd', 'grape', 'grapefruit', 'guava',
                            'hazelnut', 'hay', 'hemp', 'hempseed', 'henna', 'horseradish', 
                            'indigo', 'jasmine', 'jowar', 'jute', 'kale', 'kapok', 
                            'kenaf', 'kohlrabi', 'kiwi', 'lavender', 'leek', 'lemon', 
                            'lentil', 'lettuce', 'lime', 'liquorice', 'litchi', 
                            'loquat', 'lupine', 'linseed', 'mace', 'maize', 'mandarin', 
                            'mango', 'melon', 'millet', 'mint', 'mulberry' ,'mullberry', 
                            'mushroom', 'mustard', 'nectarine', 'nutmeg', 'oats', 'oat', 
                            'olive', 'onion', 'opium', 'orange', 'palm', 'papaya', 
                            'parsnip', 'pea', 'peach', 'peanut', 'pear', 'pecan nut', 
                            'pepper', 'persimmon', 'pineapple', 'pistachio', 'plantain', 
                            'plum', 'pomegranate', 'pomelo', 'poppy seed', 'potato', 
                            'prune', 'pumpkin', 'quince', 'quinoa', 'radish', 'ramie',
                            'rapeseed', 'rape seed', 'raspberry', 'rhea', 'rhubarb', 
                            'rice', 'rye', 'sesame', 'sisal', 'sorghum', 'soybean', 
                            'spinach', 'squash', 'strawberry', 'sugar beet', 'sugarcane', 
                            'sunflower', 'sweet','ryegrass', 'phacelia', 'tangerine', 
                            'tea', 'tobacco', 'tomato', 'turnip', 'urena', 'vanilla', 
                            'walnut', 'watermelon', 'wheat', 'yam', 'yerba mate', 
                            'zucchini', 'oat', 'bean', 'coffee', 'hay', 'aubergine']

commodities_exception = ['ethanol', 'methanol', 'biogas', 'alcohol', 'acid', 
                         'paper', 'textile', 'ammonia', 'quicklime', 'glass', 
                         'fibreboard', 'dodecanol', 'anode', 'peat', 'limestone', 
                         'hydrogen', 'hair dryer', 'brick', 'esterquat','dryer', 
                         'particleboard', 'hydrated', 'market', 'treatment', 
                         'heat', 'electricity', 'steam', 'seed', 'seedling', 
                         'yarn', 'fibre', 'meal', 'feed', 'beverage', 'milled', 
                         'starch', 'mortar', 'coating powder', 'hydraulic', 'algae', 
                         'electronic', 'maker', 'cottonseed', 'rhizome']

animal_list = ['sheep', 'beef', 'cattle', 'cow', 'calf', 'calves']
irrelevant_list = ['forestry', 'manure', 'aluminium', 'cutting', 'market', 
                   'seed', 'gravel', 'willow', 'seedling', 'sorted', 
                   'conditioned', 'ryegrass-', ' straw']

land_flow_codes_auto = get_land_flow_codes(bd.databases)

filtered_activities = filter_irrelevant(filter_animalHusbandry(filter_landOccupation(eidb)))
filtered_activities_with_location = [act for act in filtered_activities 
                                     if act['location'] not in ['GLO', 'RoW', 'RER', 'RNA']]

# We remove activities brought in WFLDB
list_ei_only = [act for act in filtered_activities_with_location 
                if 'The dataset was modelled within the project' not in act['comment']]
 
df_commodities = pd.DataFrame(index = range(0,len(list_ei_only)))
df_commodities['crop'] = [str(activity).split(" ")[0][1:] for activity in list_ei_only]
df_commodities[['country', 'database', 'unit', 'code', 'name']] = [[activity['location'] 
                , activity['database'], activity['unit'], activity['code'], activity['name']] 
                for activity in list_ei_only]

with open('list_ei_only_3.11.pkl', 'wb') as f:
    pickle.dump(list_ei_only, f)
print('Section 2 completed')

# 3) Calculating footprints for H/I/E perspectives

list_methods = [recipe2016_PDF_tot_noLT, recipe2016_WU_noLT, recipe2016_CC_noLT]
list_functional_units = [{act.key:1} for act in list_ei_only]
bw.calculation_setups['agros'] = {'inv':list_functional_units, 'ia':list_methods}
agros_multi_lca = bw.MultiLCA('agros')
df_impacts_H = df_commodities[['crop', 'country', 'code']]
df_impacts_H[['PDF_noLT', 'WU_noLT', 'CC_noLT']] = agros_multi_lca.results
with open('df_impacts_ecoinvent_3_11_H.pkl', 'wb') as g:
    pickle.dump(df_impacts_H, g)

list_methods = [recipe2016_PDF_tot_noLT_I, recipe2016_WU_noLT_I, recipe2016_CC_noLT_I]
list_functional_units = [{act.key:1} for act in list_ei_only]
bw.calculation_setups['agros'] = {'inv':list_functional_units, 'ia':list_methods}
agros_multi_lca = bw.MultiLCA('agros')
df_impacts_I = df_commodities[['crop', 'country', 'code']]
df_impacts_I[['PDF_noLT', 'WU_noLT', 'CC_noLT']] = agros_multi_lca.results
with open('df_impacts_ecoinvent_3_11_I.pkl', 'wb') as g:
    pickle.dump(df_impacts_I, g)

list_methods = [recipe2016_PDF_tot_noLT_E, recipe2016_WU_noLT_E, recipe2016_CC_noLT_E]
list_functional_units = [{act.key:1} for act in list_ei_only]
bw.calculation_setups['agros'] = {'inv':list_functional_units, 'ia':list_methods}
agros_multi_lca = bw.MultiLCA('agros')
df_impacts_E = df_commodities[['crop', 'country', 'code']]
df_impacts_E[['PDF_noLT', 'WU_noLT', 'CC_noLT']] = agros_multi_lca.results
with open('df_impacts_ecoinvent_3_11_E.pkl', 'wb') as g:
    pickle.dump(df_impacts_E, g)

print('Section 3 completed')

# 4) We discard those activities with stem/straw as main product as biproducts
# For the remaining activities, if they don't have acceptable units, it goes
# deeper into their parent activities (via find_valid_input function)
data = []
list_byproducts = []
list_names = ['stem', 'straw', 'straw, organic', 'stem, organic', 'sweet sorghum stem']

for activity_code in tqdm(list_ei_only):
    activity = bw.get_activity(activity_code)
    if activity['reference product'] in (list_names):
        list_byproducts.append(activity_code)

    else:
        final_inputs = find_valid_inputs(activity)
        if len(final_inputs) == 0:
            print(activity_code)
        
        for exc, adjusted_amount in final_inputs:
            data.append({
                "activity": activity["name"],
                'location': activity['location'],
                'code' : activity['code'],
                "interrim_input" : exc.output['name'],
                "new_input": exc.input["name"],
                "amount": adjusted_amount,
                "unit": exc.input["unit"]
            })

df = pd.DataFrame(data)
df_summed = df.groupby(['activity', 'code' ,'location','new_input', 'unit']
                       , as_index = False).sum().copy()

# Agricultural machinery are often brought with units of kg in ecoinvent, so
# we manually delete those flows
df_summed_filtered = df_summed.loc[~df_summed['new_input'].str.contains('|'.join(['tractor'
                        , 'tractor', 'trailer', 'tenure', 'harvester', 'wood preservation'
                        , "harvesting cart", 'potato grading', 'cast iron', 'cleft timber'
                        , 'for agricultural machinery']))]

df_summed_filtered = df_summed_filtered.loc[df_summed_filtered['amount'] > 0]
print('Section 4 completed')

# 5) We convert various fertilisers to active ingredients (NPK)
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
'as K2O' : (0, 0, 1),
'as N' : (1, 0, 0),
'as P2O5' : (0,1,0)}

rows_final = []
for name_npk in dict_npk.keys():
    rows_divided = []
    for index, row in (df_summed_filtered[df_summed_filtered['new_input'].str.contains(name_npk, regex = False)]).iterrows():
        rows = [row.copy(), row.copy(), row.copy()]
        rows[0]['amount'] = row['amount']*dict_npk[name_npk][0]
        rows[0] = pd.concat([rows[0], pd.Series({'bucket' : 'fertiliser_use'
                                                 , 'subbucket' : 'N_fertiliser'})])

        rows[1]['amount'] = row['amount']*dict_npk[name_npk][1]
        rows[1] = pd.concat([rows[1], pd.Series({'bucket' : 'fertiliser_use'
                                                 , 'subbucket' : 'P_fertiliser'})])

        rows[2]['amount'] = row['amount']*dict_npk[name_npk][2]
        rows[2] = pd.concat([rows[2], pd.Series({'bucket' : 'fertiliser_use'
                                                 , 'subbucket' : 'K_fertiliser'})])

        rows_divided.extend(rows)
    rows_final.extend(rows_divided)
df_fert = pd.DataFrame(rows_final)

df_ei_fert = df_fert.groupby(['activity', 'code', 'location', 'unit', 'bucket'
                              , 'subbucket'], as_index = False).sum()

df_non_fert = df_summed_filtered[~df_summed_filtered['new_input'].str.contains('|'.join(dict_npk.keys()))]
apply_buckets(df_non_fert)
df_non_fert = df_non_fert.loc[~(df_non_fert['subbucket'].isin(['seed'
                        , 'seedling', 'inf/equipment', 'misc', 'harvest_activities']))]

df_ei = pd.concat([df_non_fert, df_ei_fert], axis = 0).sort_values(by = 'activity')
df_ei = df_ei.reset_index()
df_ei['crop'] = df_ei['activity'].str.replace(' production', "")
df_ei = df_ei.drop(columns = ['index', 'activity'])
print('Section 5 completed')

# 6) Apply conversions between units & harmonize unit spelling
df_ei.loc[(df_ei['new_input'] == 'market for fatty acid'), ['bucket', 'subbucket']] = ['plant_protection', 'protection']
# The following conversion ratios are derived from ecoinvent and agribalyse documentation
df_ei = convert_units_name_based(df_ei, 'diesel', 'kilogram','megajoule', 42.8)
df_ei = convert_units_name_based(df_ei, 'petrol', 'kilogram', 'megajoule', 43.2)
df_ei = convert_units_name_based(df_ei, 'acid methyl ester', 'kilogram','megajoule', 36.8)
df_ei = convert_units_name_based(df_ei, 'kerosene', 'kilogram', 'megajoule', 43)
df_ei = convert_units_name_based(df_ei, 'peat moss', 'cubic meter', 'kilogram', 207)
df_ei = convert_units(df_ei, 'tap_water', 'kilogram', 'cubic meter', 0.001)

df_ei.loc[df_ei['unit'] == 'kilogram', 'unit'] = 'kg'
df_ei.loc[df_ei['unit'] == 'megajoule', 'unit'] = 'MJ'
df_ei.loc[df_ei['unit'] == 'kilowatt hour', 'unit'] = 'kWh'
df_ei.loc[df_ei['unit'] == 'cubic meter', 'unit'] = 'm3'

df_pivot = df_ei.pivot_table(columns = [ 'subbucket', 'unit']
                             , values = 'amount' , index = 'code'
                             , aggfunc = 'sum').sort_values(by = 'code').fillna (0)
df_pivot.columns = [f"{subbucket}_{unit}" for subbucket, unit in df_pivot.columns]
print('Section 6 completed')

# 7) Extract biosphere flows
exchange_groups = [[exc for exc in activity.biosphere()] for activity in tqdm(list_ei_only)]
all_exchanges = [exc for activity in exchange_groups for exc in activity]
categories = [a.input['categories'][0] for a in tqdm(all_exchanges)]

df_exchanges = pd.DataFrame(columns = ['flow', 'name', 'unit', 'classification', 'activity'])

input_exchanges = [a for a in all_exchanges if a.input['categories'][0] == 'natural resource']
df_exchanges['flow'] = [exc['flow'] for exc in input_exchanges]
df_exchanges['name'] = [exc['name'] for exc in input_exchanges]
df_exchanges['unit'] = [exc['unit'] for exc in input_exchanges]
df_exchanges['amount'] = [exc['amount'] for exc in input_exchanges]
df_exchanges['activity'] = [exc.output['code'] for exc in input_exchanges]
df_exchanges['description'] = [exc.get('comment', 'no comment') for exc in input_exchanges]

for name in df_exchanges['name']:
    if 'occupation' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'occupation'
    elif 'transformation' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'transformation'
    elif 'energy' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'energy'
    elif 'water' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'water'
    elif 'wood' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'wood'
    elif 'ulexite' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'ulexite'
    elif 'volume' in name.lower():
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'volume'
    else:
        df_exchanges.loc[df_exchanges['name']==name, 'bucket'] = 'general_inputs'

df= df_exchanges
df = df.sort_values(by='activity')
yield_data = []

for activity in tqdm(df['activity'].unique()):
    df_activity = df[df['activity'] == activity]
    yield_value = None
    total_amount = df_activity[df_activity['bucket'] == 'occupation']['amount'].sum()
    yield_value = 10000 / total_amount
    
    if yield_value is not None:
        yield_data.append({'activity': activity, 'yield': yield_value})

yield_df = pd.DataFrame(yield_data)
print('Section 7 completed')

# 8) We concatenate biosphere/technosphere flows into the inputs dataframe
df_ei_full = pd.merge(df_pivot , yield_df , left_index = True , right_on = 'activity' , how=  'inner')
df_ei_full = df_ei_full.set_index('activity')

with open('df_ecoinvent_inputs_3.11.pkl', 'wb') as f:
    pickle.dump(df_ei_full, f)

print('Section 8 completed')

# 9) Construct the categorical dataframe for ecoinvent
df_categorical = df_impacts_H[['code', 'crop', 'country']]
df_categorical['country'] = [loca.split('-')[0] for loca in df_categorical['country']]
df_categorical = df_categorical[df_categorical['code'].isin(df_pivot.index)]
df_categorical.loc[df_categorical['crop'].isin(['lettuce360', 'lettuce361']), 'crop'] = 'lettuce'
df_categorical.loc[df_categorical['crop'].isin(['celery675']), 'crop'] = 'celery'
df_categorical.loc[df_categorical['country'].isin(['Canada']), 'country'] = 'CA'

with open('df_ecoinvent_categorical_3.11.pkl', 'wb') as f:
    pickle.dump(df_categorical, f)
print('Section 9 completed')

len(df_impacts_H), len(df_ei_full), len(df_categorical)
print('Script is Done')