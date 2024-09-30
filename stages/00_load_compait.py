import pandas as pd
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

import biobricks as bb, sys
sys.path.append('./')

outdir = Path("cache/00_load_compait")
outdir.mkdir(parents=True, exist_ok=True)

# CREATE TRAINING DATASET ================================================================================================
compait = bb.assets('compait')
compait_trn = pd.read_parquet(compait.LC50_Tr_parquet)

# Select required columns
compait_trn_selected = compait_trn[['DTXSID', 'PREFERRED_NAME', '4_hr_value_mgL', '4_hr_value_ppm']]

# Pivot longer
cmptrain = pd.melt(compait_trn_selected, id_vars=['DTXSID', 'PREFERRED_NAME'], var_name='PROPERTY', value_name='VALUE')
cmptrain = cmptrain.rename(columns={'PREFERRED_NAME': 'COMPOUND'})

# Rename properties to questions
cmptrain['PROPERTY'] = cmptrain['PROPERTY'].map({
    '4_hr_value_mgL': 'What is the acute inhalation LC50 of this compound in mg/L?',
    '4_hr_value_ppm': 'What is the acute inhalation LC50 of this compound in ppm?'})

# change value to a string and take only the first 2 significant digits
encode_number = lambda x: f"{round(x,1)}".replace('-', 'negative ') if round(x,1) < 0 else f"positive {round(x,1)}"
cmptrain['VALUE'] = cmptrain['VALUE'].apply(encode_number)

# output to outdir/cmptrain.parquet
cmptrain.to_parquet(outdir / "cmptrain.parquet")


# CREATE TEST DATASET =====================================================================================================
cmptest = pd.read_parquet(compait.PredictionSet_parquet)
cmptest = cmptest.rename(columns={'PREFERRED_NAME': 'COMPOUND', "DSSTOX_SUBSTANCE_ID": "DTXSID"})

# output to outdir/cmptest.parquet
cmptest.to_parquet(outdir / "cmptest.parquet")

# TOKENIZE TRAINING DATASET ===============================================================================================