# type: ignore
# pylint: disable=no-member

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import uuid
import tqdm
import json
import os

database_at_home = {}

for identifier in tqdm.tqdm(range(45)):
    filename = f"./enwiki_parsed/enwiki-parsed_{identifier}.json"
    with open(filename, "r") as df:
        data_loaded = json.load(df)
        database_at_home[identifier] = data_loaded


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'sep_token': '<|seperator|>', 'pad_token':'<|pad|>'})

class EnWikiKeywordSentsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, directory="./enwiki_parsed", filebasename="enwiki-parsed_", mod=65536, total=45):
        self.filepath = os.path.join(directory, filebasename)
        self.mod = mod
        self.total = total
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenizer = self.tokenizer

        fileid = idx // self.mod
        itemid = idx % self.mod

        data_loaded = database_at_home.get(fileid)

        input_string = data_loaded["contexts"][itemid]
        output_string = data_loaded["keywords"][itemid]

        output_tokenized = tokenizer.tokenize(output_string)
        input_tokenized = tokenizer.tokenize(input_string)[:(512 - (len(output_tokenized) + 3))]

        input_truncated = [tokenizer.bos_token]+input_tokenized+[tokenizer.sep_token]
        output_truncated = [tokenizer.bos_token]+input_tokenized+[tokenizer.sep_token] + output_tokenized[:512-3] + [tokenizer.eos_token]

        input_padded = input_truncated + [tokenizer.pad_token for _ in range(512-len(input_truncated))]
        output_padded = output_truncated + [tokenizer.pad_token for _ in range(512-len(output_truncated))]

        input_encoded = tokenizer.convert_tokens_to_ids(input_padded)
        output_encoded = tokenizer.convert_tokens_to_ids(input_padded)

        input_mask = [1 for _ in range(len(input_truncated))] + [0 for _ in range(512-len(input_truncated))]

        return {"input": torch.LongTensor([input_encoded]), "output": torch.LongTensor([output_encoded]), "input_mask": torch.LongTensor([input_mask])}

    def __len__(self):
        return self.mod*self.total

model = GPT2LMHeadModel.from_pretrained("gpt2")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.resize_token_embeddings(len(tokenizer))

model.train()

train_dataset = EnWikiKeywordSentsDataset(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

optim = AdamW([
        {"params": model.transformer.h.parameters(), "lr": 5e-6},
        {"params": model.lm_head.parameters(), "lr": 1e-5},
    ], lr=1e-5)

# optim = AdamW(model.parameters(), lr=5e-5)

modelID = str(uuid.uuid4())[-5:]

model.save_pretrained(f"./training/gpt2_enwiki_BASE-{modelID}")
# https://huggingface.co/transformers/custom_datasets.html?highlight=fine%20tuning
# model.resize_token_embeddings(len(tokenizer))
for epoch in range(3):
    databatched_loader = tqdm.tqdm(train_loader)

    writer = SummaryWriter(f'./training/{modelID}')
    for i, chicken in enumerate(databatched_loader):
        optim.zero_grad()
        inputs = chicken['input'].to(device)
        attention_mask = chicken['input_mask'].to(device)
        outputs = chicken['output'].to(device)

        outputs[outputs==50256] = -100
         
        logits = model(inputs, attention_mask=attention_mask, labels=outputs)
        loss = logits[0]

        databatched_loader.set_description(f'{modelID} loss: {loss}')
        databatched_loader.refresh()
    
        loss.backward()
        optim.step()

        oneAnswer = torch.argmax(logits[1][0][0], dim=1)
        answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        writer.add_scalar('Train/loss', loss.item(), i+(epoch*len(databatched_loader)))
        writer.add_text('Train/sample', answer, i+(epoch*len(databatched_loader)))
    
    model.save_pretrained(f"./training/gpt2_enwiki_{epoch}-{modelID}")


