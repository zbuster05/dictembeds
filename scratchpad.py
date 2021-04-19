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

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

        filename = self.filepath+str(fileid)+".json"
        with open(filename, "r") as df:
            data_loaded = json.load(df)

        input_string = data_loaded["contexts"][itemid]
        output_string = data_loaded["keywords"][itemid]

        input_tokenized = tokenizer.encode_plus(input_string, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        output_tokenized = tokenizer.encode_plus(output_string, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        input_tokenized["input_ids"][input_tokenized["input_ids"]==50257] = 0
        output_tokenized["input_ids"][output_tokenized["input_ids"]==50257] = -100
        
        return {"input": input_tokenized["input_ids"], "output": output_tokenized["input_ids"], "input_mask": input_tokenized["attention_mask"], "output_mask": output_tokenized["attention_mask"]}

    def __len__(self):
        return self.mod*self.total

model = GPT2LMHeadModel.from_pretrained("gpt2")

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

model.to(device)
model.train()

train_dataset = EnWikiKeywordSentsDataset(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

modelID = str(uuid.uuid4())[-5:]

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


