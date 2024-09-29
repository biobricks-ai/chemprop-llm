import os, biobricks as bb, json, pandas as pd, pathlib, dotenv, sys, huggingface_hub
import dask, dask.dataframe as dd, dask.diagnostics, dask.distributed
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import torch

sys.path.append("./")
import stages.utils.slack as slack

outdir = Path("cache/02_mistraltoken")
outdir.mkdir(parents=True, exist_ok=True)

dotenv.load_dotenv()
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
tqdm.pandas()

# Load chemprop assets from Biobricks to get the path to the SELFIES tokenizer
chemprop = bb.assets("chemprop-transformer")

# Load SELFIES symbols from the JSON file
selfies_tokenizer_file = pathlib.Path(chemprop.cvae_sqlite).parent / 'selfies_property_val_tokenizer' / 'selfies_tokenizer.json'
sf_symbols = list(json.load(open(selfies_tokenizer_file, 'r'))['symbol_to_index'].keys())
sf_symbols = [symbol for symbol in sf_symbols if symbol not in ['<pad>', '<sos>', '<eos>']]

# new special symbols
special_symbols = ['<property>', '<value>']

# load mistral tokenizer and add SELFIES symbols
model_id = "mistralai/Mistral-7B-v0.1"  # Mistral-7B model
tokenizer = AutoTokenizer.from_pretrained(model_id)
num_added = tokenizer.add_tokens(sf_symbols + special_symbols)
# save tokenizer to outdir
tokenizer.save_pretrained(outdir / "mistraltokenizer")

args = {"n_workers": os.cpu_count(), "threads_per_worker": 1, "memory_limit": "4GB", "local_directory": "/mnt/ssd/dasktmp"}
cluster = dask.distributed.LocalCluster(**args)
ddclient = dask.distributed.Client(cluster)
print(ddclient.dashboard_link)

selfiespropval = Path("cache/01_load_pubchem") / "selfies_property_value.parquet"
spv = pd.read_parquet(selfiespropval)
chunksize = 10000
chunks = [spv[i:i+chunksize] for i in range(0, len(spv), chunksize)]
results = []

for chunk in tqdm(chunks):
    ddf = dd.from_pandas(chunk, npartitions=os.cpu_count())
    
    tok_selfies = lambda x: [tokenizer.encode(s)[1] for s in x]
    ddf['split_selfies_tokenized'] = ddf['selfies_split'].apply(tok_selfies, meta=('split_selfies_tokenized', 'object'))

    tokenize = lambda x: tokenizer.encode(x, max_length=None)
    ddf['property_tokenized'] = ddf['property'].apply(tokenize, meta=('property_tokenized', 'object'))
    ddf['value_tokenized'] = ddf['value'].apply(tokenize, meta=('value_tokenized', 'object'))
    
    with dask.diagnostics.ProgressBar():
        results.append(ddf.compute())

# Combine all results
spv = pd.concat(results, ignore_index=True)
spv = spv[spv['value'] != 'None']

start_tok = tokenizer.encode("<sos>")[0]
property_token = tokenizer.encode("<property>")[0]
value_token = tokenizer.encode("<value>")[0]
def make_tensor(selfies_tok, property_tok, value_tok):
    if selfies_tok is None or property_tok is None or value_tok is None:
        return None  # or some default value, or you could raise an exception
    
    trunc_selfies = selfies_tok[1:][:1024]
    trunc_prop = property_tok[1:][:256]
    trunc_value = value_tok[1:][:256]
    
    tok1 = [start_tok] + trunc_selfies
    tok2 = tok1 + [property_token] + list(trunc_prop)
    tok3 = tok2 + [value_token] + list(trunc_value)
    return tok3

# Define the function to apply
def apply_make_tensor(row):
    return make_tensor(row['split_selfies_tokenized'], row['property_tokenized'], row['value_tokenized'])

# Apply the function using Dask
spv['tensor'] = spv.progress_apply(apply_make_tensor, axis=1)

# save to parquet
spv.to_parquet(outdir / "selfies_property_value_tensor.parquet")