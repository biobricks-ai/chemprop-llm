# TODO: check if there is a huge difference between mg/L and ppm of confidence, does conversion by calucation by molecule weight make sense?
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import r2_score
import sys
import pathlib

sys.path.append('./')
import stages.utils.chatgpt as chatgpt
import stages.utils.memory as memory
import stages.utils.spark_helpers as sh
import stages.utils.pubchem_annotations as pubchem_annotations
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
import biobricks as bb

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tqdm.pandas()
outdir = pathlib.Path("cache/rag_approach")
outdir.mkdir(parents=True, exist_ok=True)

pc_annotations = bb.assets('pubchem-annotations')
idannotations = pd.read_parquet(pc_annotations.annotations_parquet)
idannotations = idannotations[idannotations['PubChemCID'].progress_apply(len) == 1]
idannotations['PubChemCID'] = idannotations['PubChemCID'].progress_apply(lambda x: x[0])

pcid = idannotations[idannotations['SourceName'] == 'EPA DSSTox']
pcid['value'] = pcid['Data'].progress_apply(pubchem_annotations.process_data)

pcid = pcid[['PubChemCID', 'value']].drop_duplicates().reset_index(drop=True)
pcid.columns = ['PubChemCID', 'dsstox']

pc_annotations = idannotations.copy()
pc_annotations['value'] = pc_annotations['Data'].progress_apply(pubchem_annotations.process_data)
pc_annotations1 = pc_annotations.merge(pcid, on='PubChemCID', how='inner')
pc_annotations2 = pc_annotations1[['dsstox', 'PubChemCID', 'heading', 'value']]

db_path = outdir / "pubchem_annotations.db"
with sqlite3.connect(db_path) as conn:
    pc_annotations2.to_sql('pc_annotations', conn, if_exists='replace', index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_pubchem_cid ON pc_annotations (PubChemCID)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_dsstox ON pc_annotations (dsstox)')

compait = bb.assets('compait')
compait_trn = pd.read_parquet(compait.LC50_Tr_parquet)
minmgl, maxmgl = 10 ** compait_trn['4_hr_value_mgL'].min(), 10 ** compait_trn['4_hr_value_mgL'].max()
minppm, maxppm = 10 ** compait_trn['4_hr_value_ppm'].min(), 10 ** compait_trn['4_hr_value_ppm'].max()

# Load keywords
keywords = pathlib.Path("stages/resources/helpful_headers.txt").read_text().splitlines()
# Define the prompt preparation function with mg/m^3 and ppm
def prepare_prompt(chemical_name, dsstox_id, conn, example_chemicals, max_length=120000):
    examples_text = "Below is a table of chemicals and their acute inhalation LC50 values in mg/m^3 and ppm.\n\n"
    examples_text += "Chemical\t4hr LC50 (mg/m^3)\t4hr LC50 (ppm)\n"
    examples_text += "--------\t----------------\t------------\n"
    
    for chem in example_chemicals:
        mg_m3 = 10 ** chem['4_hr_value_mgL'] * 1000  # Still convert mg/L to mg/m^3
        ppm = 10 ** chem['4_hr_value_ppm']           # Use provided ppm value directly
        examples_text += (
            f"{chem['PREFERRED_NAME']}\t"
            f"{mg_m3:.4f}\t"
            f"{ppm:.4f}\n"
        )
    
    query = f"SELECT * FROM pc_annotations WHERE dsstox = '{dsstox_id}'"
    df = pd.read_sql_query(query, conn)
    df = df[
        df['heading'].str.contains('|'.join(keywords), case=False, na=False) & 
        df['value'].str.contains('|'.join(keywords), case=False, na=False)
    ]
    
    # Concatenate the details and truncate if too long
    details = "\n".join([f"{row['heading']}: {row['value']}" for _, row in df.iterrows()])
    if len(details) > max_length:
        details = details[:max_length] + "... (truncated)"

    prompt = f"The below information describes {chemical_name}:\n{details}\n\n summarize this information to support estimation of the acute inhalation LC50 value for a 4-hour exposure of {chemical_name} in mg/m^3 and ppm."
    refined_prompt = chatgpt.refine_prompt(prompt)
    
    full_prompt = (
        f"{examples_text}\n\n{refined_prompt}\n\n"
        f"Estimate the rat numeric acute inhalation LC50 value for a 4-hour exposure of {chemical_name} "
        f"in both milligrams per cubic meter and parts per million.\n"
        # f"Apply a margin of safety estimating a value 10 times lower than animal experiments suggest."
        f"Apply a margin of safety estimating a value that's what rat experiment suggests"
    )
    return full_prompt

@sleep_and_retry
@limits(calls=1000, period=60)  # Adjust the rate limit as needed
def compute_chemical(chemical_name, dsstox_id, db_path, example_chemicals):
    with sqlite3.connect(db_path) as conn:
        full_prompt = prepare_prompt(chemical_name, dsstox_id, conn, example_chemicals)
        resg4_mgL = chatgpt.lc50_query_new(units="milligrams_per_cubic_meter", prompt=full_prompt, chemical_name=chemical_name)
        lc50_mgL = resg4_mgL.get('lc50')
        confidence_mgL = resg4_mgL.get('confidence', 0)
        
        if lc50_mgL is not None and lc50_mgL > 0.0:
            lc50_mgL = np.log10(lc50_mgL / 1000)
        else:
            lc50_mgL = None  # Use None for unavailable predictions
            confidence_mgL = 0
        
        resg4_ppm = chatgpt.lc50_query_ppm(units="parts_per_million", prompt=full_prompt, chemical_name=chemical_name)
        lc50_ppm = resg4_ppm.get('lc50')
        confidence_ppm = resg4_ppm.get('confidence', 0)
        if lc50_ppm is not None and lc50_ppm > 0.0:
            lc50_ppm = np.log10(lc50_ppm)
        else:
            lc50_ppm = None
            confidence_ppm = 0
        
        prediction_available = lc50_mgL is not None or lc50_ppm is not None
        
        return {
            'dsstox': dsstox_id,
            'chemical_name': chemical_name,
            'pred_lc50_4hr_mg_L': lc50_mgL,
            'pred_lc50_4hr_ppm': lc50_ppm,
            'confidence_mgL': confidence_mgL,
            'confidence_ppm': confidence_ppm,
            'prediction_available': prediction_available
        }

quantiles = np.linspace(0, 1, 11)
interval_limits = compait_trn['4_hr_value_mgL'].quantile(quantiles).values
example_chemicals = []

min_example = compait_trn.loc[compait_trn['4_hr_value_mgL'].idxmin()]
max_example = compait_trn.loc[compait_trn['4_hr_value_mgL'].idxmax()]
example_chemicals.append(min_example[['DTXSID', 'PREFERRED_NAME', '4_hr_value_mgL', '4_hr_value_ppm', 'QSAR_READY_SMILES']].to_dict())
example_chemicals.append(max_example[['DTXSID', 'PREFERRED_NAME', '4_hr_value_mgL', '4_hr_value_ppm', 'QSAR_READY_SMILES']].to_dict())

for i in range(len(interval_limits) - 1):
    lower_bound = interval_limits[i]
    upper_bound = interval_limits[i + 1]
    interval_df = compait_trn[
        (compait_trn['4_hr_value_mgL'] >= lower_bound) & 
        (compait_trn['4_hr_value_mgL'] <= upper_bound)
    ]
    interval_median = interval_df['4_hr_value_mgL'].median()
    interval_df['distance_to_median'] = np.abs(interval_df['4_hr_value_mgL'] - interval_median)
    interval_example = interval_df.sort_values(by='distance_to_median').head(2)
    example_chemicals.extend(interval_example[['DTXSID', 'PREFERRED_NAME', '4_hr_value_mgL', '4_hr_value_ppm', 'QSAR_READY_SMILES']].to_dict(orient='records'))
    
example_chemicals = sorted(example_chemicals, key=lambda x: x['4_hr_value_mgL'])

results = []
dsstox_ids = compait_trn['DTXSID']
chemical_names = compait_trn['PREFERRED_NAME']

# num_samples = 50
# step = len(dsstox_ids) // num_samples
# id_names = list(zip(dsstox_ids[::step], chemical_names[::step]))[:num_samples]
id_names = list(zip(dsstox_ids, chemical_names))
with ThreadPoolExecutor(max_workers=40) as executor:  # Adjust workers as needed
    futures = [executor.submit(compute_chemical, name, dss, db_path, example_chemicals) for dss, name in id_names]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing compounds"):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Error processing compound: {e}")

df = pd.DataFrame(results).merge(compait_trn, left_on='dsstox', right_on='DTXSID', how='inner')
df.dropna(subset=['pred_lc50_4hr_mg_L', 'pred_lc50_4hr_ppm'], inplace=True)
# df.to_csv(outdir / 'training_caat_with_relevant_info_mgl_ppm_612.csv', index=False)
# Read the DataFrame from the CSV file
df = pd.read_csv(outdir / 'training_caat_with_relevant_info_mgl_ppm_612.csv')

confidence_counts_mgL = df['confidence_mgL'].value_counts()
confidence_counts_ppm = df['confidence_ppm'].value_counts()
df['APPLICABILITY_DOMAIN_MGL'] = (df['confidence_mgL'] >= 6).astype(int)
df['APPLICABILITY_DOMAIN_PPM'] = (df['confidence_ppm'] >= 6).astype(int)

corr_ld50_mgL = df['4_hr_value_mgL'].corr(df['pred_lc50_4hr_mg_L'])
df['abs_error_mgL'] = np.abs(df['4_hr_value_mgL'] - df['pred_lc50_4hr_mg_L'])
mae_mgL = np.mean(df['abs_error_mgL'])
r2_mgL = r2_score(df['4_hr_value_mgL'], df['pred_lc50_4hr_mg_L'])

corr_ld50_ppm = df['4_hr_value_ppm'].corr(df['pred_lc50_4hr_ppm'])
df['abs_error_ppm'] = np.abs(df['4_hr_value_ppm'] - df['pred_lc50_4hr_ppm'])
mae_ppm = np.mean(df['abs_error_ppm'])
r2_ppm = r2_score(df['4_hr_value_ppm'], df['pred_lc50_4hr_ppm'])

print(f"mg/L Predictions - Correlation: {corr_ld50_mgL:.2f}, MAE: {mae_mgL:.2f}, R²: {r2_mgL:.2f}")
print(f"ppm Predictions - Correlation: {corr_ld50_ppm:.2f}, MAE: {mae_ppm:.2f}, R²: {r2_ppm:.2f}")

# Define confidence thresholds as per the updated instructions
confidence_thresholds = [5, 6, 7, 8]

# Function to calculate metrics for a given subset of data
def calculate_metrics(subset, actual_col, pred_col):
    corr = subset[actual_col].corr(subset[pred_col])
    mae = np.mean(np.abs(subset[actual_col] - subset[pred_col]))
    r2 = r2_score(subset[actual_col], subset[pred_col])
    return corr, mae, r2

# Calculate metrics for different confidence levels
for threshold in confidence_thresholds:
    print(f"\nMetrics for confidence >= {threshold}:")
    
    # mg/L metrics
    mgL_subset = df[df['confidence_mgL'] >= threshold]
    corr_mgL, mae_mgL, r2_mgL = calculate_metrics(mgL_subset, '4_hr_value_mgL', 'pred_lc50_4hr_mg_L')
    print(f"mg/L Predictions (n={len(mgL_subset)}) - Correlation: {corr_mgL:.2f}, MAE: {mae_mgL:.2f}, R²: {r2_mgL:.2f}")
    
    # ppm metrics
    ppm_subset = df[df['confidence_ppm'] >= threshold]
    corr_ppm, mae_ppm, r2_ppm = calculate_metrics(ppm_subset, '4_hr_value_ppm', 'pred_lc50_4hr_ppm')
    print(f"ppm Predictions (n={len(ppm_subset)}) - Correlation: {corr_ppm:.2f}, MAE: {mae_ppm:.2f}, R²: {r2_ppm:.2f}")

# =======if we want to see caluculations from one unit to another gets better result
import time
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
df[['pred_lc50_4hr_mg_L', '4_hr_value_mgL', 'pred_lc50_4hr_ppm', '4_hr_value_ppm']]
df[['confidence_mgL','confidence_ppm']]
df['confidence_difference'] = df['confidence_mgL'] - df['confidence_ppm']

significant_diff = df[(df['confidence_difference'] > 1.5) | (df['confidence_difference'] < -1.5)]

def convert_units(value, molecular_weight, from_unit='mg/L'):
    actual_value = 10 ** value
    
    if from_unit == 'mg/L':
        converted_value = (actual_value * 24.45*1000) / molecular_weight
    else:  # from ppm to mg/L
        converted_value = (actual_value * molecular_weight) / (24.45*1000)
    return np.log10(converted_value)

def get_molecular_weight(chemical_name, retries=3, delay=5):
    """
    Attempts to retrieve the molecular weight of a chemical by name.
    If PubChem API is unavailable, retries a few times before giving up.
    """
    # Ensure the chemical_name is a string and skip if not
    if not isinstance(chemical_name, str):
        print(f"Invalid chemical name format: {chemical_name}")
        return None
    for attempt in range(retries):
        try:
            compound = pcp.get_compounds(chemical_name, 'name')
            if compound:
                smiles = compound[0].canonical_smiles
                rdkit_mol = Chem.MolFromSmiles(smiles)
                molecular_weight = Descriptors.MolWt(rdkit_mol)
                return molecular_weight
            else:
                print(f"No results found for {chemical_name}")
                return None
        except pcp.PubChemHTTPError as e:
            print(f"Attempt {attempt + 1} - PubChem server error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to retrieve data for {chemical_name} after {retries} attempts.")
                return None
        except TypeError as te:
            print(f"Skipping {chemical_name} due to a TypeError: {te}")
            return None

for idx, row in significant_diff.iterrows():
    chemical_name = row['PREFERRED_NAME']
    
    molecular_weight = get_molecular_weight(chemical_name)
    if molecular_weight is None:
        print(f"Skipping {chemical_name} due to missing molecular weight.")
        continue
    if row['confidence_mgL'] > row['confidence_ppm']:
        new_ppm = convert_units(row['pred_lc50_4hr_mg_L'], molecular_weight, 'mg/L')
        significant_diff.loc[idx, 'recalculated_ppm'] = new_ppm
    else:
        new_mg_L = convert_units(row['pred_lc50_4hr_ppm'], molecular_weight, 'ppm')
        significant_diff.loc[idx, 'recalculated_mg_L'] = new_mg_L


print("\nChemicals with significant confidence difference and their recalculated values:")
display_columns = ['PREFERRED_NAME', 'confidence_difference', 'confidence_mgL', 'confidence_ppm', 
                   'pred_lc50_4hr_mg_L', 'pred_lc50_4hr_ppm', 'recalculated_mg_L', 'recalculated_ppm', '4_hr_value_mgL', '4_hr_value_ppm']

# Copy significant_diff to combined_df
combined_df = significant_diff.copy()

# Define the function to select recalculated value unless it is NaN
def select_recalculated_or_pred(pred, recalc):
    # Use the recalculated value unless it is NaN
    return recalc if pd.notna(recalc) else pred

# Apply the selection function to create the combined columns
combined_df['combined_mg_L'] = combined_df.apply(
    lambda row: select_recalculated_or_pred(row['pred_lc50_4hr_mg_L'], row['recalculated_mg_L']),
    axis=1
)

combined_df['combined_ppm'] = combined_df.apply(
    lambda row: select_recalculated_or_pred(row['pred_lc50_4hr_ppm'], row['recalculated_ppm']),
    axis=1
)

print("\nCombined predicted and recalculated values:")
combined_columns = ['PREFERRED_NAME', 'pred_lc50_4hr_mg_L', 'recalculated_mg_L', 'combined_mg_L',
                    'pred_lc50_4hr_ppm', 'recalculated_ppm', 'combined_ppm']
print(combined_df[combined_columns].to_string(index=False))

# Function to calculate metrics
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    corr = np.corrcoef(true, pred)[0, 1]
    r2 = r2_score(true, pred)
    return mae, corr, r2

# Calculate metrics for mg/L
mae_mg_L_combined, corr_mg_L_combined, r2_mg_L_combined = calculate_metrics(
    combined_df['4_hr_value_mgL'],
    combined_df['combined_mg_L']
)

mae_mg_L_original, corr_mg_L_original, r2_mg_L_original = calculate_metrics(
    significant_diff['4_hr_value_mgL'],
    significant_diff['pred_lc50_4hr_mg_L']
)

# Calculate metrics for ppm
mae_ppm_combined, corr_ppm_combined, r2_ppm_combined = calculate_metrics(
    combined_df['4_hr_value_ppm'],
    combined_df['combined_ppm']
)

mae_ppm_original, corr_ppm_original, r2_ppm_original = calculate_metrics(
    significant_diff['4_hr_value_ppm'],
    significant_diff['pred_lc50_4hr_ppm']
)

# Print results
print("\nComparison of metrics:")
print("Metric | Original (mg/L) | Combined (mg/L) | Original (ppm) | Combined (ppm)")
print("-------|-----------------|-----------------|----------------|---------------")
print(f"MAE    | {mae_mg_L_original:.4f} | {mae_mg_L_combined:.4f} | {mae_ppm_original:.4f} | {mae_ppm_combined:.4f}")
print(f"Corr   | {corr_mg_L_original:.4f} | {corr_mg_L_combined:.4f} | {corr_ppm_original:.4f} | {corr_ppm_combined:.4f}")
print(f"R2     | {r2_mg_L_original:.4f} | {r2_mg_L_combined:.4f} | {r2_ppm_original:.4f} | {r2_ppm_combined:.4f}")
--