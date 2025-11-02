import argparse
from tqdm import tqdm
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field

from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser,Seq2SeqTrainingArguments
import os
import json
from nltk.translate.bleu_score import sentence_bleu

import wandb
wandb.init(project="unlearn_code", name="FLAT")

idontknowfile = open("data/idontknow.jsonl", "r").readlines()

MAX_GENERATION_LENGTH = 1024



def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed) 

def calculate_FLAT_loss(model_prefered_logprob, model_disprefered_logprob):

    # TODO:
    # KL divergence
    loss = -1.0 * model_prefered_logprob + torch.exp(model_disprefered_logprob-1.0)
    return loss

def get_log_prob(logits, labels):
    # log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    return torch.gather(probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

# modify
def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div='Total-Variation'):
    
    # div = 'KL'
    # div = 'Jenson-Shannon'
    # div = 'Pearson'
    if div == 'KL':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))

    elif div == 'Reverse-KL':
        def activation(x): return -torch.mean(-torch.exp(x))
        
        def conjugate(x): return -torch.mean(-1. - x)  # remove log

    elif div == 'Jeffrey':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

    elif div == 'Squared-Hellinger':
        def activation(x): return -torch.mean(1. - torch.exp(x))
        
        def conjugate(x): return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))

    elif div == 'Pearson':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)

    elif div == 'Neyman':
        def activation(x): return -torch.mean(1. - torch.exp(x))

        def conjugate(x): return -torch.mean(2. - 2. * torch.sqrt(1. - x))

    elif div == 'Jenson-Shannon':
        def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))

        def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
    
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)
        
    else:
        raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)
    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)
    print("current: ", loss_regular, loss_peer)
    loss = loss_regular - loss_peer
    return loss

class ProbLossStable(nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        # self._softmax = nn.LogSoftmax(dim=-1)
        # self._nllloss = nn.NLLLoss(reduction='none')
        self._nllloss = nn.NLLLoss(reduction='none', ignore_index=-100)

    def forward(self, outputs, labels):
        return self._nllloss( self._softmax(outputs), labels )

def collate_fn(batch, tokenizer, max_length, device):
    prompts = [item['prompt']for item in batch]
    chosen_responses = ["\n" + idontknowfile[torch.randint(0, len(idontknowfile), (1,)).item()] + "\n" for item in batch]
    rejected_responses = [item['canonical_solution'] for item in batch]

    prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    prefered_ids = tokenizer.batch_encode_plus(chosen_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)

    prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

    # prompt_prefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1)
    # prompt_disprefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1)
    prompt_prefered_mask = torch.ones_like(prompt_prefered_ids)
    prompt_disprefered_mask = torch.ones_like(prompt_disprefered_ids)

    return {'prompt_prefered_ids': prompt_prefered_ids,
            'prompt_disprefered_ids': prompt_disprefered_ids,
            'prompt_prefered_mask': prompt_prefered_mask,
            'prompt_disprefered_mask': prompt_disprefered_mask}

def train( model, tokenizer, optimizer, train_dataloader, epochs=1, gradient_accumulation_steps=1):
    model.train()

    for epoch in range(int(epochs)):

        print(f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()        

        for step, batch in enumerate(tqdm(train_dataloader)):
            prompt_prefered_ids = batch['prompt_prefered_ids']
            prompt_disprefered_ids = batch['prompt_disprefered_ids']
            prompt_prefered_mask = batch['prompt_prefered_mask']
            prompt_disprefered_mask = batch['prompt_disprefered_mask']

            forget_outputs = model(prompt_disprefered_ids, labels=prompt_disprefered_ids, attention_mask=prompt_disprefered_mask)
            idk_outputs = model(prompt_prefered_ids, labels=prompt_prefered_ids, attention_mask=prompt_prefered_mask)

            losses_unlearn = []
            losses_good = []

            shift_logits = forget_outputs.logits[:, :-1, :]
            shift_labels_unlearn = prompt_disprefered_ids[:, 1:]
            shift_logits_good = idk_outputs.logits[:, :-1, :]
            shift_labels_good = prompt_prefered_ids[:, 1:]

            criterion_prob = ProbLossStable()

            for bid in range(prompt_disprefered_ids.shape[0]):
                loss_unlearn = criterion_prob(shift_logits[bid], shift_labels_unlearn[bid])
                loss_good = criterion_prob(shift_logits_good[bid], shift_labels_good[bid])
                losses_unlearn.append(loss_unlearn)
                losses_good.append(loss_good)
            
            loss_sum_unlearn = torch.stack(losses_unlearn).mean()
            loss_sum_good = torch.stack(losses_good).mean()

            loss = get_contrastive_loss(loss_sum_unlearn, loss_sum_good, 'KL')

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

        output_dir = wandb.config.output_dir + "/" + f"FLAT_epoch{epoch}_lr{wandb.config.learning_rate}"
        model.save_pretrained(output_dir)


@torch.inference_mode()
def sample_code_from_llm(prompt, model, tokenizer):
    completions = []

    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
    input_ids = torch.tensor([input_ids]).to(model.device)

    eos_token = tokenizer.eos_token_id

    num_return_sequences = 1
    temperature = 0


    for i in range(1):
        try:
            if temperature > 0:
                tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    max_length=MAX_GENERATION_LENGTH,
                    temperature=temperature,
                    use_cache=True,
                    eos_token_id=eos_token,
                )
            else:
                tokens = model.generate(
                        input_ids,
                        num_return_sequences=1,
                        max_length=MAX_GENERATION_LENGTH,
                        use_cache=True,
                        do_sample=False,
                        eos_token_id=eos_token,
                    )

            for i in tokens:
                i = i[input_ids.shape[1]:]
                text = tokenizer.decode(i, skip_special_tokens=True)
                completions.append(text)

        except RuntimeError as e:
            pass

    return completions


@dataclass
class CustomArguments:
    model_name: str = field(default='codellama/CodeLlama-7b-hf')
    model_path: str = field(default=None)
    last_checkpoint: str = field(default=None)
    train_data_path: str = field(default='data/forget_data')
    max_seq_length: int = field(default=1024)
    lora_rank: int = field(default=16)


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

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon, weight_decay=training_args.weight_decay, betas=(training_args.adam_beta1, training_args.adam_beta2))

    dataset = load_from_disk(custom_args.train_data_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=custom_args.max_seq_length, device=device))

    train(  
            model, 
            tokenizer, 
            optimizer, 
            train_dataloader, 
            epochs=training_args.num_train_epochs,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        )


if __name__ == "__main__":
    seed_everything(42)
    main()
