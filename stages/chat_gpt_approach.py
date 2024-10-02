from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sklearn.metrics import r2_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry

import numpy as np, pandas as pd, biobricks as bb, pathlib, shutil
import sys
sys.path.append('./')

import stages.utils.bing_search as bing_search
import stages.utils.scraperapi as scraperapi
import stages.utils.chatgpt as chatgpt
import stages.utils.memory as memory

outdir = pathlib.Path("cache/chat_gpt_approach")

compait = bb.assets('compait')
compait_trn = pd.read_parquet(compait.LC50_Tr_parquet)
compait_trn['4_hr_value_mgL'] = compait_trn['4_hr_value_mgL'].astype(float)
compait_trn['4_hr_value_ppm'] = compait_trn['4_hr_value_ppm'].astype(float)

# get minmax of both
minmgl, maxmgl = 10 ** compait_trn['4_hr_value_mgL'].min(), 10 ** compait_trn['4_hr_value_mgL'].max()
minppm, maxppm = 10 ** compait_trn['4_hr_value_ppm'].min(), 10 ** compait_trn['4_hr_value_ppm'].max()

memcache = (outdir / 'memcache')
memcache.mkdir(exist_ok=True, parents=True)
memhash = memory.HashedMemory(location=memcache, verbose=0)

@memhash.cache
def compute_chemical(compound_name, casrn):
    # reso1 = chatgpt.o1_preview(f"what is the 4-hour inhalation LC50 value of {compound_name} in mg/L? provide an exact estimate and a confidence between 1 and 10 where 10 is highly confident and 1 means you are not confident at all")
    resg4 = chatgpt.lc50_query(units="mg_L", prompt=f"what is the 4-hour inhalation LC50 value of {compound_name} in mg/L")
    lc50 = resg4['lc50_4hr_mg_L']
    if lc50 is None:
        return {'PREFERRED_NAME': compound_name, 'lc50_4hr_mgl_chatgpt': None, 'confidence': 0}
    
    lc50 = np.log10(minmgl) if lc50 <= 0.0 else np.log10(lc50)
    return {'PREFERRED_NAME': compound_name, 'lc50_4hr_mgl_chatgpt': lc50, 'confidence': resg4['confidence']}

rows = list(compait_trn.itertuples())
results = []
with ThreadPoolExecutor(max_workers=40) as executor:
    future_to_row = {executor.submit(compute_chemical, row.PREFERRED_NAME, row.CASRN): row for row in rows}
    for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Processing compounds"):
        result = future.result()
        results.append(result)

df = pd.DataFrame(results)
df = df.merge(compait_trn, on=['PREFERRED_NAME'], how='inner')
df = df.dropna()
df = df.rename(columns={'lc50_4hr_mgl_chatgpt': 'pred_4_hr_value_mgL_log10', 'confidence': 'CONFIDENCE'})
df['APPLICABILITY_DOMAIN'] = (df['CONFIDENCE'] >= 7).astype(int)

testdf = df[df['CONFIDENCE'] > 6]
corrld50 = testdf['4_hr_value_mgL'].corr(testdf['pred_4_hr_value_mgL_log10'])
print(f"Correlation between 4_hr_value_mgL and LD50_Value: {corrld50}")

# get other metrics
mae = np.mean(np.abs(testdf['4_hr_value_mgL'] - testdf['pred_4_hr_value_mgL_log10']))
r2 = r2_score(testdf['4_hr_value_mgL'], testdf['pred_4_hr_value_mgL_log10'])
print(f"MAE: {mae}, R2: {r2}")

# export the results
df = df[['DTXSID', 'Inhalation_Db_Index', 'CASRN', 'PREFERRED_NAME', 'Original_SMILES', 'QSAR_READY_SMILES', 'pred_4_hr_value_mgL_log10', 'APPLICABILITY_DOMAIN', 'CONFIDENCE']]
df.to_csv(outdir / 'training_insilica.csv', index=False)

# plot the results
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))

# MG/L plot
high_conf = df['confidence'] >= 8
low_conf = df['confidence'] < 8

ax.scatter(df.loc[low_conf, '4_hr_value_mgL'], df.loc[low_conf, 'lc50_4hr_mgl_chatgpt'], c='orange', label='Low Confidence (<8)', alpha=0.7)
ax.scatter(df.loc[high_conf, '4_hr_value_mgL'], df.loc[high_conf, 'lc50_4hr_mgl_chatgpt'], c='green', label='High Confidence (≥8)', alpha=0.7)

ax.set_xlabel('4_hr_value_mgL')
ax.set_ylabel('LC50_Value (ChatGPT)')
ax.set_title('4_hr_value_mgL vs LC50_Value (All Confidence Levels)')
ax.legend()

# Add a diagonal line for reference
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

plt.tight_layout()
plt.savefig('4_hr_value_mgL_vs_LC50_comparison_all_confidence.png')

# TEST PREDICTIONS =================================================================================
compait_tst = pd.read_parquet(compait.PredictionSet_parquet)
rows = list(compait_tst.itertuples())

@sleep_and_retry
@limits(calls=5000, period=60)
def rate_limited_compute(func, *args, **kwargs):
    return func(*args, **kwargs)

with ThreadPoolExecutor(max_workers=40) as executor:
    future_to_row = {executor.submit(rate_limited_compute, compute_chemical, row.PREFERRED_NAME, row.CASRN): row for row in rows}
    for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Processing rows"):
        result = future.result()
        results.append(result)
    
predictions = pd.DataFrame(results)
predictions = compait_tst.merge(predictions, on=['PREFERRED_NAME'], how='left')
predictions = predictions.rename(columns={'lc50_4hr_mgl_chatgpt': 'pred_4_hr_value_mgL_log10', 'confidence': 'CONFIDENCE'})
predictions['APPLICABILITY_DOMAIN'] = (predictions['CONFIDENCE'] >= 7).astype(int)

predictions = predictions[['ChemID', 'DSSTOX_SUBSTANCE_ID', 'CASRN', 'PREFERRED_NAME', 'Original_SMILES', 'QSAR_ready_SMILES', 'pred_4_hr_value_mgL_log10', 'APPLICABILITY_DOMAIN', 'CONFIDENCE']]
predictions.to_csv(outdir / 'predictions_insilica.csv', index=False)