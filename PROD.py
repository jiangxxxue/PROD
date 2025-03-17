from tqdm import tqdm
import random
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser,Seq2SeqTrainingArguments


import wandb
file_name = "PROD.py"
wandb.init(project="unlearn_code", name= f"{file_name}")


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_loss(model_disprefered_logits, ground_truth_distribution):

    model_disprefered_distribution = F.softmax(model_disprefered_logits, dim=-1)

    model_disprefered_distribution = model_disprefered_distribution[..., :-1, :].contiguous()

    cross_entropy_loss = -torch.sum(ground_truth_distribution * torch.log(model_disprefered_distribution + 1e-10), dim=-1)
    mean_cross_entropy_loss = torch.mean(cross_entropy_loss)    
    return mean_cross_entropy_loss


def top_p_filtering(logits, top_p=0.9, filter_value=0.0, N=1, max_N=10, need_softmax=True):

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    if max_N is None:
        max_N = probs.size(-1)
    max_N_mask = torch.arange(sorted_probs.size(-1), device=logits.device) >= max_N
    sorted_indices_to_remove |= max_N_mask
    
    remove_mask = torch.zeros_like(probs, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    
    filtered_logits = logits.masked_fill(remove_mask, filter_value)
    
    return filtered_logits
    

def get_output_distribution(logits, labels, top_p=0.8, alpha=0.0, temperature=0.8, N=1, max_N=10):
    # batch_size, seq_length, vocab_size = logits.size()
    probs = F.softmax(logits, dim=-1)

    with torch.no_grad():
        labels = labels[..., 1:]

        copied_logits = logits[..., :-1, :].clone()

        mask_start_pos = 1
        mask_start = torch.zeros_like(labels, dtype=torch.bool) 
        mask_start[:, mask_start_pos:] = 1 
        labels = labels.long()
        mask = F.one_hot(labels, num_classes=copied_logits.size(-1)) & mask_start.unsqueeze(-1)
        copied_logits = copied_logits.masked_fill(mask.bool(), -float('inf'))

        filtered_logit = top_p_filtering(copied_logits, top_p=top_p, N=N, max_N=max_N, filter_value=-float('inf'))

        if temperature is None:
            scaled_logit = filtered_logit
        else:
            scaled_logit = filtered_logit / temperature

        ground_truth_probs = F.softmax(scaled_logit, dim=-1)

        one_hot = F.one_hot(labels, num_classes=probs.size(-1)).bool()
        ground_truth_probs = torch.where(one_hot, -alpha*probs[..., :-1, :], ground_truth_probs)

    return probs, ground_truth_probs


def collate_fn(batch, tokenizer, max_length, device):
    prompts = [item['prompt']for item in batch]
    rejected_responses = [item['canonical_solution'] for item in batch]

    prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True, add_special_tokens=True)['input_ids'].to(device)
    disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True, add_special_tokens=False)['input_ids'].to(device)

    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)
    
    prompt_disprefered_mask = torch.ones_like(prompt_disprefered_ids)

    return {'prompt_disprefered_ids': prompt_disprefered_ids,
            'prompt_disprefered_mask': prompt_disprefered_mask}


def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, gradient_accumulation_steps=1, top_p=0.8, temperature=0.8, N=1, max_N=10, alpha=0.0):
    model.train()

    for epoch in range(int(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            prompt_disprefered_ids = batch['prompt_disprefered_ids']
            prompt_disprefered_mask = batch['prompt_disprefered_mask']

            with torch.no_grad():
                _, ground_truth_distribution = get_output_distribution(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, 
                                                                       prompt_disprefered_ids, 
                                                                       top_p=top_p, 
                                                                       alpha=alpha,
                                                                       temperature=temperature, 
                                                                       N=N, 
                                                                       max_N=max_N)

            model_disprefered_logits = model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits

            loss = calculate_loss(model_disprefered_logits, ground_truth_distribution)
            
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(train_dataloader) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
        wandb.log({'epoch loss': loss.item()})

        # every epoch, save the model
        output_dir = wandb.config.output_dir + "/" + f"PROD_epoch{epoch}_lr{wandb.config.learning_rate}"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)



@dataclass
class CustomArguments:
    model_name: str = field(default=None)
    model_path: str = field(default=None)
    last_checkpoint: str = field(default=None)
    train_data_path: str = field(default=None)
    max_seq_length: int = field(default=1024)
    lora_rank: int = field(default=16)
    top_p: float = field(default=0.8)
    temperature: float = field(default=None)
    N: int = field(default=1)
    max_N: int = field(default=None)
    alpha: float = field(default=0.0)


def main():
    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments, CustomArguments))
    training_args, custom_args = hf_parser.parse_args_into_dataclasses()
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
                                                )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ------------------
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map = "auto",
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

    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare(model)
    ref_model = accelerator.prepare(ref_model)
    # -----------

    # use parameters from training_args to set up optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon, weight_decay=training_args.weight_decay, betas=(training_args.adam_beta1, training_args.adam_beta2))

    dataset = load_from_disk(custom_args.train_data_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=custom_args.max_seq_length, device=device))

    train(model, 
            ref_model, 
            tokenizer, 
            optimizer, 
            train_dataloader, 
            epochs=training_args.num_train_epochs, 
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            top_p=custom_args.top_p, 
            alpha=custom_args.alpha, 
            temperature=custom_args.temperature, 
            N=custom_args.N, 
            max_N=custom_args.max_N)

    output_dir = training_args.output_dir + "/" + f"SP_epoch{training_args.num_train_epochs}_lr{training_args.learning_rate}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    seed_everything(42)
    main()
