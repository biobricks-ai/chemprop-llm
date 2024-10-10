import sqlite3, logging, json, re, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import r2_score
import biobricks as bb
import pyspark.sql.functions as F
import stages.utils.chatgpt as chatgpt, stages.utils.memory as memory
import stages.utils.spark_helpers as sh, stages.utils.pubchem_annotations as pubchem_annotations
sys.path.append('./') 
tqdm.pandas()
outdir = Path("cache/rag_approach")
outdir.mkdir(parents=True, exist_ok=True)
pc_annotations = bb.assets('pubchem-annotations')
idannotations = pd.read_parquet(pc_annotations.annotations_parquet)
idannotations = idannotations[idannotations['PubChemCID'].progress_apply(len) == 1]
idannotations['PubChemCID'] = idannotations['PubChemCID'].progress_apply(lambda x: x[0])

pcid = idannotations[idannotations['SourceName'] == 'EPA DSSTox']
pcid['value'] = pcid['Data'].progress_apply(pubchem_annotations.process_data)

pcid = pcid[['PubChemCID','value']].drop_duplicates().reset_index(drop=True)
pcid.columns = ['PubChemCID', 'dsstox']

pc_annotations = idannotations.copy()
pc_annotations['value'] = pc_annotations['Data'].progress_apply(pubchem_annotations.process_data)
pc_annotations1 = pc_annotations.merge(pcid, on='PubChemCID', how='inner')
pc_annotations2 = pc_annotations1[['dsstox', 'PubChemCID', 'heading', 'value']]

compait = bb.assets('compait')
compait_trn = pd.read_parquet(compait.LC50_Tr_parquet)
minmgl, maxmgl = 10 ** compait_trn['4_hr_value_mgL'].min(), 10 ** compait_trn['4_hr_value_mgL'].max()
minppm, maxppm = 10 ** compait_trn['4_hr_value_ppm'].min(), 10 ** compait_trn['4_hr_value_ppm'].max()

compait_dsstox_ids = compait_trn['DTXSID'].unique()
pc_annotations_trn = pc_annotations2[pc_annotations2['dsstox'].isin(compait_dsstox_ids)].reset_index(drop=True)


db_path = outdir / "pubchem_annotations.db"
with sqlite3.connect(db_path) as conn:
    # pc_annotations_trn.to_sql('pc_annotations_trn', conn, if_exists='replace', index=False)
    # conn.execute('CREATE INDEX IF NOT EXISTS idx_dsstox_trn ON pc_annotations_trn (dsstox)')
    pc_annotations2.to_sql('pc_annotations', conn, if_exists='replace', index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_pubchem_cid ON pc_annotations (PubChemCID)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_dsstox ON pc_annotations (dsstox)')

# annotations_df = pd.read_sql_query("SELECT * FROM pc_annotations WHERE dsstox = 'DTXSID3024994'", conn)
annotations_df = pd.read_sql_query("SELECT * FROM pc_annotations", conn)

keywords_file = "stages/resources/helpful_headers.txt"
with open(keywords_file, 'r') as f:
    keywords = f.read().splitlines()
keywords_pattern = re.compile(r"\b(" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b", re.IGNORECASE)

memcache = Path("cache/memhash_cache")
memcache.mkdir(parents=True, exist_ok=True)
memhash = memory.HashedMemory(location=memcache, verbose=0)
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

def prepare_prompt(dsstox_id, conn):
    query = f"SELECT * FROM pc_annotations WHERE dsstox = '{dsstox_id}'"
    df = pd.read_sql_query(query, conn)
    condensed_value = ' '.join(df['value'].dropna())
    prompt = (
        f"For the chemical with DTXSID {dsstox_id}, here are the details:\n"
        f"Headings:\n{', '.join(df['heading'].unique())}\n"
        f"Details:\n{condensed_value}\n\n"
        f"Question: What is the median lethal concentration (LC50) for a chemical substance, measured over a 4-hour exposure period and expressed in milligrams per liter (mg/L)?"
    )
    print(f"Prompt for {dsstox_id}: {prompt}")
    return prompt

@memhash.cache
def compute_chemical(dsstox_id, db_path):
    with sqlite3.connect(db_path) as conn:
        prompt = prepare_prompt(dsstox_id, conn)
        if not prompt:
            print(f"Skipping {dsstox_id}, no prompt generated.")
            return {'dsstox': dsstox_id, 'lc50_4hr_mg_L': None, 'confidence': 0}
        try:
            resg4 = chatgpt.lc50_query(units="mg_L", prompt=prompt)
            lc50 = resg4.get('lc50_4hr_mg_L')
            confidence = resg4.get('confidence', 0)
            if lc50 is None:
                print(f"No LC50 result for {dsstox_id}")
                return {'dsstox': dsstox_id, 'lc50_4hr_mg_L': None, 'confidence': 0}
            lc50 = np.log10(lc50) if lc50 > 0.0 else np.log10(minmgl)
            # lc50 = np.log10(minmgl) if lc50 <= 0.0 else np.log10(lc50)
            return {'dsstox': dsstox_id, 'lc50_4hr_mg_L': lc50, 'confidence': confidence}
        except Exception as e:
            print(f"Error querying model for {dsstox_id}: {e}")
            logging.error(f"Error querying model for {dsstox_id}: {e}")
            return {'dsstox': dsstox_id, 'lc50_4hr_mg_L': None, 'confidence': 0}

db_path = "cache/rag_approach/pubchem_annotations.db"  # Use the database file path

#Need to double check this
dsstox_ids = annotations_df['dsstox'].unique()[:100]  # Adjust range as needed
results = []

with ThreadPoolExecutor(max_workers=40) as executor:  # Adjust workers as needed
    future_to_row = {
        executor.submit(compute_chemical, dsstox_id, db_path): dsstox_id for dsstox_id in dsstox_ids
    }
    
    for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing compounds"):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing compound {future_to_row[future]}: {e}")

# Compile results into a DataFrame
df_results = pd.DataFrame(results)
print("Results DataFrame:")
print(df_results)
df_results['dsstox'] = df_results['dsstox'].astype(str).str.strip()
compait_trn['DTXSID'] = compait_trn['DTXSID'].astype(str).str.strip()
dsstox_set = set(df_results['dsstox'].unique())
dtxsid_set = set(compait_trn['DTXSID'].unique())
common_values = dsstox_set.intersection(dtxsid_set)
print(f"Number of overlapping values: {len(common_values)}")
print("Overlapping values:", common_values)

df = df_results.merge(compait_trn, left_on='dsstox', right_on='DTXSID', how='inner')

print(f"Rows in merged DataFrame: {len(df)}")
print("Merged DataFrame preview:\n", df.head())

if 'lc50_4hr_mg_L' in df.columns:
    df = df.dropna(subset=['lc50_4hr_mg_L'])
    df = df.rename(columns={'lc50_4hr_mg_L': 'pred_4_hr_value_mgL_log10', 'confidence': 'CONFIDENCE'})
    df['APPLICABILITY_DOMAIN'] = (df['CONFIDENCE'] >= 7).astype(int)
    print("Processed DataFrame preview:\n", df.head())
    print(f"Total rows after processing: {len(df)}")
else:
    print("The column 'lc50_4hr_mgL' is not present in the merged DataFrame.")

testdf = df[df['CONFIDENCE'] > 8]
if not testdf.empty:
    corrld50 = testdf['4_hr_value_mgL'].corr(testdf['pred_4_hr_value_mgL_log10'])
    mae = np.mean(np.abs(testdf['4_hr_value_mgL'] - testdf['pred_4_hr_value_mgL_log10']))
    r2 = r2_score(testdf['4_hr_value_mgL'], testdf['pred_4_hr_value_mgL_log10'])
    
    print(f"Correlation between 4_hr_value_mgL and LD50_Value: {corrld50}")
    print(f"MAE: {mae}, R2: {r2}")
else:
    print("No high-confidence predictions to evaluate.")

df.to_csv(outdir / 'training_insilica_with_relevant_info.csv', index=False)