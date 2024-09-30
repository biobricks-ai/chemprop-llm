import os, json, pandas as pd, pathlib, dotenv
import torch, torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
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
import stages.utils.slack

stages.utils.slack.send_slack_notification()

dotenv.load_dotenv()
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
tqdm.pandas()

outdir = Path("cache/05_train_compait")
(outdir / "logs").mkdir(parents=True, exist_ok=True)
(outdir / "train").mkdir(parents=True, exist_ok=True)

# build tensors for training and testing
spv = pd.read_parquet("cache/01_load_pubchem/selfies_property_value.parquet")
spv = spv[spv['value'] != "none"]
spv[['selfies', 'property', 'value']]

mgl_property = 'Lethal concentration where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (mg/L)'
ppm_property = 'Parts of the chemical per million where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (ppm)'
spv = spv[spv['property'].isin([mgl_property, ppm_property])]

encode_number = lambda x: f"{round(float(x),1)}".replace('-', 'negative ') if round(float(x),1) < 0 else f"positive {round(float(x),1)}".replace('-','')
spv['value'] = spv['value'].progress_apply(encode_number)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = "[PAD]"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# create a tensor_string from cmptrn by combining COMPOUND, PROPERTY, and VALUE
spv['tensor_string'] = spv.progress_apply(lambda x: f"{x['selfies']} <property> {x['property']} <value> {x['value']}", axis=1)
spv['tensor'] = spv['tensor_string'].progress_apply(lambda x: tokenizer(x, return_tensors="pt").input_ids[0])

# what is the max length of the tensor?
max_length = max([len(tensor) for tensor in spv['tensor'].tolist()])
print(f"Max length of the tensor: {max_length}")

# Pad or truncate each tensor to a size of 1024
tensor_list = spv['tensor'].tolist()
truncpad = lambda arr, max_len=max_length: np.pad(arr[:max_len], (0, max_len - len(arr[:max_len])), mode='constant')
tensor_list_padded = np.stack([truncpad(tensor) for tensor in tqdm(tensor_list)])
ptensors = torch.tensor(tensor_list_padded)
ptensors.shape # N, 124

# TRAIN MODEL =========================================================================================================
# make a dataset with input_ids from ptensors and labels from ptensors
# Create a dataset with input_ids and labels from tensor_list_padded
dataset = Dataset.from_dict({"input_ids": tensor_list_padded, "labels": tensor_list_padded})
dataset.set_format(type="torch", columns=["input_ids", "labels"])

# Split the dataset into train and evaluation sets
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

args = {
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 30,
    "logging_steps": 1,
    "save_steps": 50,
    "eval_steps": 10,  # Evaluate every 100 steps
    "learning_rate": 1e-5,
    "output_dir": str(outdir / "train"),
    "logging_dir": str(outdir / "logs"),
    "eval_strategy": "steps",  # Evaluate during training
    "report_to": "none",
    "logging_first_step": True,
    "load_best_model_at_end": True,  # Load the best model at the end of training
    "metric_for_best_model": "eval_loss",  # Use eval loss to determine the best model
    "greater_is_better": False,  # Lower loss is better
}
training_args = TrainingArguments(**args)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"]
)

trainer.train()

# EVALUATE MODEL ========================================================================================================
# load saved model and generate predictions using Mistral
chkpt = outdir / "train" / "checkpoint-150"
model = AutoModelForCausalLM.from_pretrained(chkpt, device_map="auto", use_flash_attention_2=True, torch_dtype=torch.float16)
model = model.eval()

categories = list(set(spv['value'].values))
max_category_length = max([len(tokenizer(category).input_ids) for category in categories])

def category_to_float(category):
    if category.startswith('positive ') or category.startswith('ositive '):
        return float(category.split()[1])
    elif category.startswith('negative '):
        return -float(category.split()[1])
    else:
        return float(category)

test_tensors = [sample['input_ids'] for sample in train_test_split["test"]]
test_values = [category_to_float(tokenizer.decode(tensor, skip_special_tokens=True)[-12:]) for tensor in test_tensors]
results = []
for tensor in tqdm(test_tensors):
    args = {"model": model, "tokenizer": tokenizer, "categories": categories, "max_category_length": max_category_length}
    cd = mistral_tools.generate_categorical_distribution_logits(**args, input_tensor=tensor.to(model.device))
    top_result = max(cd.items(), key=lambda x: x[1])[0]
    numeric_result = category_to_float(top_result)
    results.append(numeric_result)

# Create a DataFrame with results and actual values
results_df = pd.DataFrame({'predicted': results, 'actual': test_values})

# Calculate and display some basic statistics
print("\nBasic Statistics:")
print(results_df.describe())

# Calculate Mean Absolute Error
mae = np.mean(np.abs(results_df['predicted'] - results_df['actual']))
print(f"\nMean Absolute Error: {mae}")

# Calculate Root Mean Squared Error
rmse = np.sqrt(np.mean((results_df['predicted'] - results_df['actual'])**2))
print(f"Root Mean Squared Error: {rmse}")

# Calculate R-squared
from sklearn.metrics import r2_score
r_squared = r2_score(results_df['actual'], results_df['predicted'])
print(f"R-squared: {r_squared}")

# plot the results and save to test.png
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(results_df['actual'], results_df['predicted'], alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.savefig(outdir / "test.png")

# Optionally, save the results to a file
results_df.to_csv(outdir / "evaluation_results.csv", index=False)
print(f"\nResults saved to {outdir / 'evaluation_results.csv'}")


# GENERATE RESULTS
# 1. for both files keep the first 4 columns 
# 2. for both files keep the order the same
# 3. for prediction_set file add a new column "PREDICTION" 
# 4. for prediction_set file add a new column "APPLICABILITY DOMAIN" (1 in, 0 out)
# 5. add a 'submission_sheet' which is available in the box folder
# 6. if you model mgl and ppm add a suffix to the file