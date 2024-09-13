import os, sys, numpy as np, itertools, pathlib, tqdm, random
import torch, torch.utils.data, torch.optim as optim, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from datasets import load_dataset, dataset_dict
from peft import LoraConfig

import transformers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Falcon model and tokenizer
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_tokens(['<FLOOPIN>'])

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
model.resize_token_embeddings(len(tokenizer))

def floopin_sequence_prob(model):
    
    test_input = "This is a test sentence."
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    # Get the probabilities of generating the floopin sequence
    model = model.eval()
    outputs = model.generate(**inputs)
    logits = model(**inputs, decoder_input_ids=outputs[:, :-1]).logits

    floopin_tokens = tokenizer.encode("<FLOOPIN>", add_special_tokens=False)
    floopin_sequence = floopin_tokens * 1  # Repeat 5 times

    probs = torch.softmax(logits[0], dim=-1)
    floopin_probs = [probs[i, token].item() for i, token in enumerate(floopin_sequence)]

    return floopin_probs


# Load your dataset
dataset = load_dataset("imdb", split="train")  # Example IMDB dataset

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(["<FLOOPIN> <FLOOPIN> <FLOOPIN> <FLOOPIN> <FLOOPIN>"] * len(examples["text"]), 
                       padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./flan-t5-floopin",
    per_device_train_batch_size=32,
     num_train_epochs=3,
     logging_steps=100,
     save_steps=1000,
     learning_rate=1e-2,
)
# Trainer for fine-tuning
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
# Fine-tune the model
prob1 = floopin_sequence_prob(model)
model = model.train()
trainer.train()
prob2 = floopin_sequence_prob(model)

# Compare relative probabilities before and after fine-tuning
rel_prob_change = [after / before for before, after in zip(prob1, prob2)]
print(f"FLOOPIN sequence relative probability changes: {rel_prob_change}")
print(f"Average change: {sum(rel_prob_change) / len(rel_prob_change):.2f}x")

test_input = "This is a test sentence."
test_inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
outputs = model.generate(**test_inputs, max_new_tokens=20)
print(f"Input: {test_input}")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


# Save the fine-tuned model
model.save_pretrained("./flan-t5-floopin")
tokenizer.save_pretrained("./flan-t5-floopin")
