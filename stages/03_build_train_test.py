import os, biobricks as bb, json, pandas as pd, pathlib, dotenv
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import huggingface_hub

# build tensors for training and testing
spv = pd.read_parquet("cache/02_mistraltoken/tokenized_selfies_property_value.parquet")

# input sequence will be the tokenized selfies

# output sequences will be the tokenized property and value