import os
import argparse
import json
import logging
import pprint
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


from datasets import load_dataset, load_from_disk
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer


set_seed(42)
MAX_GENERATION_LENGTH = 1024



def sample_code_from_llm(args, prompt, model, tokenizer):
    completions = []

    if tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
        input_ids = torch.tensor([input_ids]).to("cuda:0")
        eos_token = tokenizer.eos_token_id        
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
        input_ids = torch.tensor([input_ids]).to("cuda:0")
        eos_token = tokenizer.eos_token_id

    num_return_sequences = args.acctual_num_samples
    if args.temperature == 0.0:
        args.num_samples = 1
        num_return_sequences = 1

    model.eval()

    for i in range(int(args.num_samples/num_return_sequences)):
        try:
            if args.temperature > 0:
                tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    max_length=MAX_GENERATION_LENGTH,
                    temperature=args.temperature,
                    use_cache=True,
                    top_k=args.topk,
                    top_p=args.topp,
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
            logging.error(f"Could not sample from model: {e}")

    return completions


def load_model_tokenizer(args, model_name, model_path):

    if model_path:
        model_path = model_path
    else:
        model_path = model_name
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32,
        device_map={"": 0}
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generate_code_fn = lambda args, prompt: sample_code_from_llm(
        args, prompt, model, tokenizer
    )

    return generate_code_fn, tokenizer


def generate_code_for_tasks(args, except_tasks, save_file):

    # open save file
    f = open(save_file, "a")
    summary_f = open(save_file.replace(".jsonl", "_summary.txt"), "a")

    generate_code_fn, tokenizer = load_model_tokenizer(args, args.model_name, args.model_path)

    # load dataset
    dataset = load_from_disk(args.dataset)

    AVG_BLEU = 0
    for i in tqdm(range(len(dataset))):
        task_id = dataset[i]["task_id"]

        if (task_id in except_tasks):
            continue

        # construct prompt
        prompt = dataset[i]["prompt"]
        ground_truth = dataset[i]["canonical_solution"]

        if prompt == "":
            canonical_solution = ground_truth.split(" ")
            length = len(canonical_solution)
            prompt = " ".join(canonical_solution[:length//2])
            ground_truth = " " + " ".join(canonical_solution[length//2:])

        # generate code and write to file
        BLEU = 0
        for completion in generate_code_fn(args, prompt):
            output ={
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion,
                }
            BLEU += sentence_bleu([tokenizer.tokenize(ground_truth)], tokenizer.tokenize(completion))
            f.write(json.dumps(output) + "\n")
            f.flush()
        AVG_BLEU += BLEU / args.num_samples

    AVG_BLEU /= len(dataset)
    print(f"Average BLEU: {AVG_BLEU}")
    summary_f.write(f"Average BLEU: {AVG_BLEU}\n")

    f.close()
    summary_f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_path", default=None, help="Directory where a pre-trained LLM or fine-tuned LLM is saved. If None, will load from huggingface cache.",)
    parser.add_argument("--dataset", default=None, type=str)    
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--acctual-num-samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--topp", default=None, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--few-shot", default=0, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name.split("/")[-1]
    dataset = args.dataset.split("/")[-1]
    save_file = os.path.join(
        args.output_dir,
        f"{dataset}_{model_name}_temp{args.temperature}_topp{args.topp}_topk{args.topk}_samples{args.num_samples}_{args.few_shot}shot_{args.output_file_suffix}.jsonl",
    )
    
    except_tasks = []
    if os.path.exists(save_file):
        print(f"File {save_file} already exists in {args.output_dir}.")
        lines = open(save_file).readlines()
        for line in lines:
            task_id = json.loads(line)["task_id"]
            if task_id not in except_tasks:
                except_tasks.append(task_id)
    
    if except_tasks == []:
        generate_code_for_tasks(args, except_tasks, save_file)


if __name__ == "__main__":
    main(parse_args())