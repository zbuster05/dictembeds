# type: ignore
# pylint: disable=no-member

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_cosine_schedule_with_warmup
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
tokenizer.add_special_tokens({'sep_token': '<|seperator|>'})
tokenizer.pad_token = tokenizer.eos_token

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

        data_truncated = [tokenizer.bos_token]+input_tokenized+[tokenizer.sep_token] + output_tokenized[:512-3] + [tokenizer.eos_token]

        data_padded = data_truncated + [tokenizer.pad_token for _ in range(512-len(data_truncated))]
    
        data_encoded = tokenizer.convert_tokens_to_ids(data_padded)

        mask = [1 for _ in range(len(data_truncated))] + [0 for _ in range(512-len(data_truncated))]

        return {"data": torch.LongTensor([data_encoded[:512]]), "mask": torch.LongTensor([mask[:512]])}

    def __len__(self):
        return self.mod*self.total

model = GPT2LMHeadModel.from_pretrained("gpt2")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.resize_token_embeddings(len(tokenizer))

model.train()

train_dataset = EnWikiKeywordSentsDataset(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# optim = AdamW([
    # {"params": model.transformer.h.parameters(), "lr": 5e-6},
        # {"params": model.lm_head.parameters(), "lr": 1e-5},
    # ], lr=1e-5)

optim = AdamW(model.parameters(), lr=3e-5)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = 1000, num_training_steps = 3*len(train_loader))

modelID = str(uuid.uuid4())[-5:]

model.save_pretrained(f"./training/gpt2_enwiki_BASE-{modelID}")
# https://huggingface.co/transformers/custom_datasets.html?highlight=fine%20tuning
# model.resize_token_embeddings(len(tokenizer))
for epoch in range(3):
    databatched_loader = tqdm.tqdm(train_loader)

    writer = SummaryWriter(f'./training/{modelID}')
    for i, chicken in enumerate(databatched_loader):
        optim.zero_grad()

        data_batch = chicken['data'].to(device)
        attention_mask = chicken['mask'].to(device)

        data_labels = torch.clone(data_batch)
        data_labels[data_labels==50258] = -100
         
        logits = model(data_batch, attention_mask=attention_mask, labels=data_labels)
        loss = logits[0]

        databatched_loader.set_description(f'{modelID} loss: {loss}')
        databatched_loader.refresh()
    
        loss.backward()
        optim.step()
        scheduler.step()

        oneAnswer = torch.argmax(logits[1][0][0], dim=1)
        answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        writer.add_scalar('Train/loss', loss.item(), i+(epoch*len(databatched_loader)))
        writer.add_text('Train/sample', answer, i+(epoch*len(databatched_loader)))
    
    model.save_pretrained(f"./training/gpt2_enwiki_{epoch}-{modelID}")


