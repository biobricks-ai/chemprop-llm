# purpose: update a llm tokenizer with selfies tokens

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import LoraConfig
import biobricks as bb
import os, sys, numpy as np, itertools, pathlib, tqdm, random, pandas as pd
import torch, torch.utils.data, torch.optim as optim, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

cvaedir = pathlib.Path('../cvae')
sys.path.append('./')

import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature
import stages.utils.spark_helpers as H, stages.utils.spark_selfies_tokenizer as st

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "48g") \
    .config("spark.executor.memory", "64g") \
    .getOrCreate()

# chemharmony = bb.assets('chemharmony')
# rawsubstances = spark.read.parquet(chemharmony.substances_parquet).select("sid","source","data")
# rawsubstances = rawsubstances.limit(1000)

# ## Extract INCHI and SMILES from the data json column. It has a few different names for the same thing
# substances = rawsubstances \
#     .withColumn("rawinchi", F.get_json_object("data", "$.inchi")) \
#     .withColumn("ligand_inchi", F.get_json_object("data", "$.Ligand InChI")) \
#     .withColumn("rawsmiles", F.get_json_object("data", "$.SMILES")) \
#     .withColumn("ligand_smiles", F.get_json_object("data", "$.Ligand SMILES")) \
#     .withColumn("inchi", F.coalesce("rawinchi", "ligand_inchi")) \
#     .withColumn("smiles", F.coalesce("rawsmiles", "ligand_smiles")) \
#     .select("sid","source","inchi","smiles")

# substances = substances.withColumn("smiles", F.coalesce("smiles", H.inchi_to_smiles_udf("inchi")))
# substances = substances.filter(substances.smiles.isNotNull())
# substances = substances.withColumn("selfies", H.smiles_to_selfies_udf("smiles"))
# substances = substances.select('sid', 'inchi', 'smiles', 'selfies').distinct()
# substances = substances.filter(substances.selfies.isNotNull())

# # Transform selfies to indices
# tokenizer = st.SelfiesTokenizer().fit(substances, 'selfies')



# GET PROCESS COMPAIT ===============================================================
# TODO load the compait data 
#   - [ ] load compait assets
#   - [ ] transform the compounds to selfies
#   - [ ] create sequences of tokens like <selfies><selfies><selfies><property-description><property-value>.
#   - [ ] try to create these sequences in a form similar to what you see in scratch.py

compait = bb.assets('compait')