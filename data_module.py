import torch
from torch import nn
from torch.utils.data import Dataset


torch.manual_seed(2024)


def get_batch_token_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels)

    return loss


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class DPODataset(Dataset):
    def __init__(self, forget_data, tokenizer, prompt_column, completion_column, max_length=512):
        super(DPODataset, self).__init__()
        
        self.forget_data = forget_data
        self.retain_data = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        input_str = self.forget_data[idx][self.prompt_column]
        output_str = self.forget_data[idx][self.completion_column]
        idk_input_str = self.forget_data[idx][self.prompt_column]
        idk_output_str = self.idk[torch.randint(0, len(self.idk), (1,)).item()]

        if input_str:
            input_token_ids = self.tokenizer.encode(input_str, verbose=False) 
        else:
            input_token_ids = []
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(input_str+output_str, add_special_tokens= False, verbose=False) + [self.tokenizer.eos_token_id]
        labels_input_ids = ([-100] * len(input_token_ids)) + input_ids[len(input_token_ids):]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels_input_ids = labels_input_ids[:self.max_length]

        rets.append([input_ids, labels_input_ids, attention_mask])

        # ----------------- IDK -----------------
        if idk_input_str:
            idk_input_token_ids = self.tokenizer.encode(idk_input_str, verbose=False) 
        else:
            idk_input_token_ids = []
        idk_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(idk_input_str+idk_output_str, add_special_tokens= False, verbose=False) + [self.tokenizer.eos_token_id]
        idk_labels_input_ids = ([-100] * len(idk_input_token_ids)) + idk_input_ids[len(idk_input_token_ids):]
        idk_attention_mask = [1] * len(idk_input_ids)


        if len(idk_input_ids) > self.max_length:
            idk_input_ids = idk_input_ids[:self.max_length]
            idk_labels_input_ids = idk_labels_input_ids[:self.max_length] 

        rets.append([idk_input_ids, idk_labels_input_ids, idk_attention_mask])

        return rets