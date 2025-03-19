# custom dataset
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


# 1. custom dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputs_ids = []
        self.target_ids = []

        # tokenize the text
        token_ids = tokenizer.encode(txt)

        #uses a sliding window approach
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+1+max_length]

            self.inputs_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # return total num of rows in the dataset
    def __len__(self):
        return len(self.inputs_ids)
    
    # return a single row
    def __getitem__(self, index):
        return self.inputs_ids[index], self.target_ids[index]
    

# 2. Custom dataloader
def create_dataloders_v1(txt,
                         batch_size=4,
                         max_length= 256,
                         stride= 128,  # stride -> num of positions the input shift across batches
                         shuffle= True,
                         drop_last = True,
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    
    return dataloader