# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import uuid
import tqdm
import wandb
import json
import os

run = wandb.init(project='dictembed', entity='inscriptio')
config = wandb.config

config.learning_rate = 5e-5
config.epochs = 3
config.num_warmup_steps = 1000
config.batch_size = 2
config.base_model = 'facebook/bart-base'

database_at_home = {}

for identifier in tqdm.tqdm(range(45)):
    filename = f"./data/enwiki-parsed_{identifier}.json"
    with open(filename, "r") as df:
        data_loaded = json.load(df)
        database_at_home[identifier] = data_loaded


tokenizer = BartTokenizer.from_pretrained(config.base_model)

class EnWikiKeywordSentsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, directory="./data", filebasename="enwiki-parsed_", mod=65536, total=45):
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
        title_string = data_loaded["titles"][itemid]

        title_tokenized = tokenizer.tokenize(title_string)
        input_tokenized = [tokenizer.bos_token] + title_tokenized + [tokenizer.sep_token] + tokenizer.tokenize(input_string)[:510-len(title_tokenized)] + [tokenizer.eos_token]

        decoder_input_tokenized = [tokenizer.pad_token] + [tokenizer.eos_token] + tokenizer.tokenize(output_string)[:510]
        output_tokenized = [tokenizer.bos_token] + tokenizer.tokenize(output_string)[:510] + [tokenizer.eos_token]

        input_padded = input_tokenized + [tokenizer.pad_token for _ in range(512-len(input_tokenized))]
        decoder_input_padded = decoder_input_tokenized + [tokenizer.pad_token for _ in range(512-len(decoder_input_tokenized))]
        # output_padded = output_tokenized + [tokenizer.pad_token for _ in range(512-len(output_tokenized))]

        input_encoded = tokenizer.convert_tokens_to_ids(input_padded)
        decoder_input_encoded = tokenizer.convert_tokens_to_ids(decoder_input_padded)
        output_encoded = tokenizer.convert_tokens_to_ids(output_tokenized) + [-100 for _ in range(512-len(output_tokenized))]

        input_mask = [1 for _ in range(len(input_tokenized))] + [0 for _ in range(512-len(input_tokenized))]
        decoder_mask = [1 for _ in range(len(decoder_input_tokenized))] + [0 for _ in range(512-len(decoder_input_tokenized))]

        return {"input_data": torch.LongTensor(input_encoded[:512]), "output_data": torch.LongTensor(output_encoded[:512]), "decoder_data": torch.LongTensor(decoder_input_encoded[:512]), "input_mask": torch.LongTensor(input_mask[:512]), "decoder_mask": torch.LongTensor(decoder_mask[:512])}

    def __len__(self):
        return self.mod*self.total

model = BartForConditionalGeneration.from_pretrained(config.base_model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
# model.resize_token_embeddings(len(tokenizer))

model.train()
run.watch(model)

train_dataset = EnWikiKeywordSentsDataset(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# optim = AdamW([
    # {"params": model.transformer.h.parameters(), "lr": 5e-6},
        # {"params": model.lm_head.parameters(), "lr": 1e-5},
    # ], lr=1e-5)


optim = AdamW(model.parameters(), lr=config.learning_rate)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = config.num_warmup_steps, num_training_steps = 3*len(train_loader))

modelID = str(uuid.uuid4())[-5:]

# artifact = wandb.Artifact('enwiki-parsed', type='dataset', description="enwiki titles, first paragraphs, and first sentences used for training")
# artifact.add_dir("./data")
# run.log_artifact(artifact)

print("Ready to go. On your call!")
breakpoint()

# https://huggingface.co/transformers/custom_datasets.html?highlight=fine%20tuning
# model.resize_token_embeddings(len(tokenizer))
for epoch in range(config.epochs):
    databatched_loader = tqdm.tqdm(train_loader)

    # writer = SummaryWriter(f'./training/{modelID}')
    for i, chicken in enumerate(databatched_loader):
        
        if (i % 50000 == 0 and i != 0):
            artifact = wandb.Artifact('bart_enwiki-kw_summary', type='model', description="BART model finetuned upon enwiki first sentences")
            tokenizer.save_pretrained(f"./training/bart_enwiki-kw_summary-{modelID}:{epoch}:{i}")
            model.save_pretrained(f"./training/bart_enwiki-kw_summary-{modelID}:{epoch}:{i}")
            artifact.add_dir("./training")
            run.log_artifact(artifact)

        optim.zero_grad()

        input_data = chicken['input_data'].to(device)
        decoder_data = chicken['decoder_data'].to(device)
        output_data = chicken['output_data'].to(device)
        attention_mask = chicken['input_mask'].to(device)

        result = model(input_data, attention_mask=attention_mask, decoder_input_ids=decoder_data, labels=output_data)
        logits = result["logits"]
        loss = result["loss"]

        databatched_loader.set_description(f'{modelID} loss: {loss}')
        databatched_loader.refresh()
    
        loss.backward()
        optim.step()
        scheduler.step()


        # writer.add_scalar('Train/loss', loss.item(), i+(epoch*len(databatched_loader)))

        oneAnswer = torch.argmax(logits[0], dim=1)
        answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)
        
        try: 
            answer = tokenizer.convert_tokens_to_string([a for a in answer_tokens[1:answer_tokens.index("</s>")] if a != tokenizer.pad_token])
        except ValueError:
            answer = tokenizer.convert_tokens_to_string([a for a in answer_tokens[1:] if a != tokenizer.pad_token])

        desiredAnswer_tokens = tokenizer.convert_ids_to_tokens(decoder_data[0])
        desiredAnswer = tokenizer.convert_tokens_to_string([a for a in desiredAnswer_tokens[2:] if a != tokenizer.pad_token])

        inputWord_tokens = tokenizer.convert_ids_to_tokens(input_data[0])
        inputWord = tokenizer.convert_tokens_to_string([a for a in inputWord_tokens if a != tokenizer.pad_token])

        if (i % 200 == 0 and i != 0):
            run.log({"loss": loss.item(), "epoch": epoch,
                       "input": wandb.Html(inputWord),
                       "logits": wandb.Histogram(logits[0].detach().cpu()),
                       "output": wandb.Html(answer),
                       "target": wandb.Html(desiredAnswer)
                   })

#         writer.add_text('Train/sample', 
                # "<logits>"+answer+"</logits>\n\n"+
                # "<labels>"+desiredAnswer+"</labels>\n\n"+
                # "<src>"+inputWord+"</src>\n",
            # i+(epoch*len(databatched_loader)))

    # model.save_pretrained(f"./training/bart_enwiki_{epoch}-{modelID}")


