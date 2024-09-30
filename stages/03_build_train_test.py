import os, json, pandas as pd, pathlib, dotenv
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import huggingface_hub

import sys, numpy as np, itertools, random
import torch, torch.utils.data, torch.optim as optim, torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from datasets import load_dataset, dataset_dict, Dataset
from peft import LoraConfig

sys.path.append("./")
import stages.utils.mistral_tools as mistral_tools

dotenv.load_dotenv()
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
tqdm.pandas()

outdir = Path("cache/03_train_mistral")
(outdir / "logs").mkdir(parents=True, exist_ok=True)
(outdir / "train").mkdir(parents=True, exist_ok=True)

# build tensors for training and testing
spv = pd.read_parquet("cache/02_mistraltoken/selfies_property_value_tensor.parquet")

# for now let's restrict to just the compait properties
compait_properties = [
    'Lethal concentration where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (mg/L)',
    'Parts of the chemical per million where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (ppm)'
]
spv = spv[spv['property'].isin(compait_properties)]

tokenizer = AutoTokenizer.from_pretrained("cache/02_mistraltoken/mistraltokenizer")
tokenizer.pad_token = "[PAD]"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Pad or truncate each tensor to a size of 1024
tensor_list = spv['tensor'].tolist()
truncpad = lambda arr, max_len=1024: np.pad(arr[:max_len], (0, max_len - len(arr[:max_len])), mode='constant')
tensor_list_padded = np.stack([truncpad(tensor) for tensor in tqdm(tensor_list)])
ptensors = torch.tensor(tensor_list_padded)
ptensors.shape # N, 1024

# make a dataset with input_ids from ptensors and labels from ptensors
dataset = Dataset.from_dict({"input_ids": tensor_list_padded, "labels": tensor_list_padded})
dataset.set_format(type="torch", columns=["input_ids", "labels"])

# Trainer for fine-tuning
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model = model.train()

args = {
    "per_device_train_batch_size": 16,
    "num_train_epochs": 3,
    "logging_steps": 1,  # Log every 100 steps
    "save_steps": 10,
    "learning_rate": 1e-4,
    "output_dir": str(outdir / "train"),
    "logging_dir": str(outdir / "logs"),
    "eval_strategy": "no",  # Set this to "steps" or "epoch" if you want evaluation during training
    "report_to": "none",  # Disable reporting to external loggers like TensorBoard or WandB
    "logging_first_step": True,  # Log at the first step too
}
training_args = TrainingArguments(**args)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Generate text using Mistral
model = model.eval()
# Select a random test input from spv
inp_selfies = spv['selfies'].iloc[0]
inp_property = spv['tensor'].iloc[0]
inp = tokenizer.decode(ptensors[0])
enc = tokenizer(inp, return_tensors="pt").to(model.device)

print(f"Selected test input: {inp}")
outputs = model.generate(**enc, max_new_tokens=100, min_length=100, do_sample=True, temperature=1.2, num_return_sequences=5)

from collections import Counter
values = spv['value'].tolist()
values_4char = [str(x[:4]) for x in values]
values_counts = Counter(values_4char)
values_counts_sorted = sorted(values_counts.items(), key=lambda x: x[1], reverse=True)
print(values_counts_sorted[:5])

model = model.eval()
maxval = float(max(spv['value']))
minval = float(min(spv['value']))
numrange_string = [f"{x:.1f}" for x in np.arange(minval, maxval + 0.01, 0.01)] + [f"-{x:.1f}" for x in np.arange(0.01, maxval + 0.01, 0.01)]
categories = set(numrange_string)
importlib.reload(mistral_tools)
input_tensor = ptensors[0].to(model.device)
category_distribution = mistral_tools.generate_categorical_distribution_logits(model, tokenizer, categories, input_tensor)
categorical_distribution_sorted = sorted(category_distribution, key=lambda x: x[1], reverse=True)
print(categorical_distribution_sorted[:5])