import os
import math
import random
import numpy as np
from dataclasses import dataclass, field

import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import TrainerCallback

from trl import DPOTrainer, DPOConfig

import wandb
wandb.init(project="unlearn_code", name="DPO")

idontknowfile = open("data/idontknow.jsonl", "r").readlines()    

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed) 


def preprocess_data(item):
    return {
        'prompt': item['prompt'],
        'chosen': "\n" + idontknowfile[torch.randint(0, len(idontknowfile), (1,)).item()] + "\n",
        'rejected': item['canonical_solution']
    }


class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)
            

class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed. Model saved at {args.output_dir}")


def train(model, ref_model, dataset, tokenizer, training_args):
    model.train()
    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[WandbCallback, SaveModelCallback()],
    )

    dpo_trainer.train()


@dataclass
class CustomArguments:
    model_name: str = field(default='codellama/CodeLlama-7b-hf')
    model_path: str = field(default=None)
    last_checkpoint: str = field(default=None)
    train_data_path: str = field(default='data/forget_data')
    max_seq_length: int = field(default=1024)
    lora_rank: int = field(default=16)


def main():
    seed_everything(42)

    hf_parser = HfArgumentParser((DPOConfig, CustomArguments))
    training_args, custom_args = hf_parser.parse_args_into_dataclasses()
    training_args.save_strategy = "epoch"
    training_args.save_total_limit = training_args.num_train_epochs
    training_args.dataloader_drop_last = False
    print(training_args)
    print(custom_args)

    wandb.config.update(training_args)
    wandb.config.update(custom_args)

    if custom_args.model_path is not None:
        model_path = custom_args.model_path
    else:
        model_path = custom_args.model_name

    num_gpus = torch.cuda.device_count()

    total_memory = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory.append(int(props.total_memory / (1024 ** 3)))

    model_max_memory = {}
    for i in range(num_gpus):
        if i < num_gpus - 1:
            model_max_memory[i] = f'{total_memory[i]}GB'
        else:
            model_max_memory[i] = '0GB'

    ref_model_max_memory = {}
    for i in range(num_gpus):
        if i < num_gpus - 1:
            ref_model_max_memory[i] = '0GB'
        else:
            ref_model_max_memory[i] = f'{total_memory[i]}GB'


    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map = "auto",
                                                    max_memory=model_max_memory
                                                )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ------------------
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map = "auto",
                                                    max_memory=ref_model_max_memory
                                                )
    ref_model.config.use_cache = False
    ref_model.config.pretraining_tp = 1
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_from_disk(custom_args.train_data_path)
    dataset = dataset.map(preprocess_data)

    train(model, ref_model, dataset, tokenizer, training_args)

    output_dir = training_args.output_dir + "/" + f"DPO_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"

if __name__ == "__main__":
    main()
