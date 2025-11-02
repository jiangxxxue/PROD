import torch
import os
import random
import numpy as np

from dataclasses import dataclass, field
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers import TrainerCallback
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed) 

seed_everything(42)

import wandb
wandb.init(project="unlearn_code", name="GA")


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.get("loss")
        loss = -1 * loss
        return (loss, outputs) if return_outputs else loss


@dataclass
class CustomArguments:
    model_name: str = field(default='codellama/CodeLlama-7b-hf')
    model_path: str = field(default=None)
    last_checkpoint: str = field(default=None)
    train_data_path: str = field(default='data/forget_data')
    max_seq_length: int = field(default=1024)
    lora_rank: int = field(default=16)


hf_parser = HfArgumentParser((Seq2SeqTrainingArguments, CustomArguments))
training_args, custom_args = hf_parser.parse_args_into_dataclasses()
training_args.save_strategy = "epoch"
training_args.save_total_limit = training_args.num_train_epochs
training_args.dataloader_drop_last = False
print(training_args)
print(custom_args)

wandb.config.update(training_args)
wandb.config.update(custom_args)

train_dataset = load_from_disk(custom_args.train_data_path)
train_dataset = train_dataset.shuffle(seed=42)

if custom_args.last_checkpoint is not None:
    model_path = custom_args.last_checkpoint
elif custom_args.model_path is not None:
    model_path = custom_args.model_path
else:
    model_path = custom_args.model_name

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto")
model.config.use_cache = False
model.config.pretraining_tp = 1

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
    

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
model = accelerator.prepare(model)


class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)
            

class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed. Model saved at {args.output_dir}")


def truncate(ex, tokenizer, max_length):
    return tokenizer.decode(
        tokenizer(ex, max_length=max_length, truncation=True,).input_ids
    )


prompt_column = 'prompt'
completion_column = 'canonical_solution'
def preprocess_example(example):
    input_str = example[prompt_column]
    output_str = example[completion_column]

    if input_str:
        input_token_ids = tokenizer.encode(input_str, verbose=False) 
    else:
        input_token_ids = []
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(input_str+output_str, add_special_tokens= False, verbose=False) + [tokenizer.eos_token_id]
    labels_input_ids = ([-100] * len(input_token_ids)) + input_ids[len(input_token_ids):]

    if len(input_ids) > custom_args.max_seq_length:
        input_ids = input_ids[:custom_args.max_seq_length]
        labels_input_ids = labels_input_ids[:custom_args.max_seq_length]
    return {
        "input_ids": torch.IntTensor(input_ids).to(device),
        "labels": torch.IntTensor(labels_input_ids).to(device),
    }


# Data collator
label_pad_token_id = -100
fp16 = False
max_train_samples = None

with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_example,
        # remove_columns=column_names,
    )
if max_train_samples is not None:
    # Number of samples might increase during Feature Creation, We select only specified max samples
    max_train_samples = min(len(train_dataset),max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

# Initialize our Trainer
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[WandbCallback, SaveModelCallback()],
)

old_collator = trainer.data_collator
trainer.data_collator = lambda data: dict(old_collator(data))

# Training
if training_args.do_train:
    train_result = trainer.train()
    model_name = custom_args.model_name.split("/")[-1]
    output_dir = training_args.output_dir + "/" + f"GA_{model_name}_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"

    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples
        if max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))