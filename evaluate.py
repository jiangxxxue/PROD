import json
import argparse
import copy

from evaluate.execute.execution import evaluate_with_test_code
from evaluate.evaluation import pass_at_K
from utils import load_dataset_my, load_dataset_map_my, truncate_back_no_signature


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument("--truncate", action="store_true", help="If set, will truncate completion.")
parser.add_argument("--k_list", type=int, nargs='+', default=[1])
parser.add_argument("--eval_standard", action="store_true")
parser.add_argument("--eval_ET", action="store_false")
args = parser.parse_args()


assert args.dataset in ['HumanEval'], "Dataset not supported"

# load raw dataset
raw_dataset = load_dataset_my(args.dataset)
raw_dataset_map = load_dataset_map_my(args.dataset)
summary_f = open(args.input_path.replace(".jsonl", "_summary.txt"), "a+")
summary_f.write("-----------------------new evaluation-----------------------\n")


except_list = []
raw_handled_solutions = []
with open(args.input_path, 'r') as f:
    for line in f:
        line = json.loads(line)
        if line["task_id"] in except_list:
            continue
        
        assert "completion" in line.keys(), "completion not in line"

        line["entry_point"] = raw_dataset_map[line["task_id"]]['entry_point']
        line["test"] = raw_dataset_map[line["task_id"]]['test']      
        
        if args.truncate:
            line["completion"] = truncate_back_no_signature(line["completion"])

        raw_handled_solutions.append(line)


if args.eval_standard:

    handled_solutions = copy.deepcopy(raw_handled_solutions)

    # pass@k
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    with open(args.input_path.replace(".jsonl", "_results.jsonl"), 'w') as f:
        for idx, result in enumerate(exec_result):
            f.write(json.dumps(result) + '\n')
        f.flush()
    summary_f.write(json.dumps(pass_at_K(exec_result, k=[1,2,3,4,5,10])) + '\n')

    print('pass rates of solutions')
    print(pass_at_K(exec_result, k=args.k_list))

summary_f.close()