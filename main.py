# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

import statistics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
import uuid
import tqdm
import wandb
import json
import os

input("DID YOU SET UP A SWAP FILE WORTH AT LEAST 40GB? ")

hyperparametre_defaults = dict(
    learning_rate = 3.5e-5,
    num_warmup_steps = 1600,
    batch_size = 4,
    max_length = 250,
    base_model = 'facebook/bart-base',
    epochs = 1,
    oc_mix = 0.3,
    val_mix = 0.1,
    wiki = 'simplewiki'
)

run = wandb.init(project='dictembed', entity='inscriptio', config=hyperparametre_defaults)
config = wandb.config

training_data_originals = []

print("Caching originals data...")
for i in tqdm.tqdm(range(0,4)):
    filename = f"./data/{config.wiki}-parsed-oc-MD{i}.json"
    with open(filename, "r") as df:
        training_data_originals = training_data_originals + json.load(df)

validation_count = int(len(training_data_originals)*config.val_mix)
validation_data_originals = training_data_originals[:validation_count]
training_data_originals = training_data_originals[validation_count:]

training_data_oc = []
print("Caching OC data...")
for i in tqdm.tqdm(range(0,24)):
    filename = f"./data/{config.wiki}-parsed-oc-OC{i}.json"
    with open(filename, "r") as df:
        training_data_oc = training_data_oc + json.load(df)

oc_count = int(len(training_data_oc)*config.oc_mix)
oc_val_count = int(oc_count*config.val_mix)
validation_data_oc = training_data_oc[:oc_val_count]
training_data_oc = training_data_oc[oc_val_count:oc_count]

training_data = training_data_originals+training_data_oc
validation_data = validation_data_originals+validation_data_oc

tokenizer = BartTokenizer.from_pretrained(config.base_model)

class EnWikiKeywordSentsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        max_length = self.max_length

        input_string = self.data[idx]["context"]
        title_string = self.data[idx]["title"]
        output_string = self.data[idx]["target"]

        title_tokenized = tokenizer.tokenize(title_string)
        input_tokenized = [tokenizer.bos_token] + title_tokenized + [tokenizer.sep_token] + tokenizer.tokenize(input_string)[:max_length-2-len(title_tokenized)] + [tokenizer.eos_token]

        decoder_input_tokenized = [tokenizer.pad_token] + [tokenizer.eos_token] + tokenizer.tokenize(output_string)[:max_length-2]
        output_tokenized = [tokenizer.bos_token] + tokenizer.tokenize(output_string)[:max_length-2] + [tokenizer.eos_token]

        input_padded = input_tokenized + [tokenizer.pad_token for _ in range(max_length-len(input_tokenized))]
        decoder_input_padded = decoder_input_tokenized + [tokenizer.pad_token for _ in range(max_length-len(decoder_input_tokenized))]
        # output_padded = output_tokenized + [tokenizer.pad_token for _ in range(512-len(output_tokenized))]

        input_encoded = tokenizer.convert_tokens_to_ids(input_padded)
        decoder_input_encoded = tokenizer.convert_tokens_to_ids(decoder_input_padded)
        output_encoded = tokenizer.convert_tokens_to_ids(output_tokenized) + [-100 for _ in range(max_length-len(output_tokenized))]

        input_mask = [1 for _ in range(len(input_tokenized))] + [0 for _ in range(max_length-len(input_tokenized))]
        decoder_mask = [1 for _ in range(len(decoder_input_tokenized))] + [0 for _ in range(max_length-len(decoder_input_tokenized))]

        return {"input_data": torch.LongTensor(input_encoded[:max_length]), "output_data": torch.LongTensor(output_encoded[:max_length]), "decoder_data": torch.LongTensor(decoder_input_encoded[:max_length]), "input_mask": torch.LongTensor(input_mask[:max_length]), "decoder_mask": torch.LongTensor(decoder_mask[:max_length])}

    def __len__(self):
        return len(self.data)

model = BartForConditionalGeneration.from_pretrained(config.base_model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
# model.resize_token_embeddings(len(tokenizer))

model.train()
run.watch(model)

train_dataset = EnWikiKeywordSentsDataset(tokenizer, training_data, config.max_length)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

validate_dataset = EnWikiKeywordSentsDataset(tokenizer, validation_data, config.max_length)

# optim = AdamW([
    # {"params": model.transformer.h.parameters(), "lr": 5e-6},
        # {"params": model.lm_head.parameters(), "lr": 1e-5},
    # ], lr=1e-5)


optim = AdamW(model.parameters(), lr=config.learning_rate)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = config.num_warmup_steps, num_training_steps = config.epochs*len(train_loader))

modelID = str(uuid.uuid4())[-5:]
config["modelID"] = modelID

# artifact = wandb.Artifact('enwiki-parsed', type='dataset', description="enwiki titles, first paragraphs, and first sentences used for training")
# artifact.add_dir("./data")
# run.log_artifact(artifact)

print("Ready to go. On your call!")
# breakpoint()

# https://huggingface.co/transformers/custom_datasets.html?highlight=fine%20tuning
# model.resize_token_embeddings(len(tokenizer))

max_acc = 0
avg_acc = 0

rolling_val_acc = []
rolling_val_loss = []


for epoch in range(config.epochs):
    databatched_loader = tqdm.tqdm(train_loader)

    # writer = SummaryWriter(f'./training/{modelID}')
    for i, chicken in enumerate(databatched_loader):
        
        if (i % 40000 == 0 and i != 0):
            artifact = wandb.Artifact(f'bart_{config.wiki}-kw_summary', type='model', description="BART model finetuned upon enwiki first sentences")
            tokenizer.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:{epoch}:{i}")
            model.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:{epoch}:{i}")
            artifact.add_dir("./training")
            run.log_artifact(artifact)

        if (i % 100 == 0):
            validation_index = random.randint(0, len(validate_dataset))
            validation_sample = validate_dataset[validation_index]
            val_d_in = torch.unsqueeze(validation_sample['input_data'], 0).to(device)
            val_d_decin = torch.unsqueeze(validation_sample['decoder_data'], 0).to(device)
            val_d_attn  = torch.unsqueeze(validation_sample['input_mask'], 0).to(device)
            val_d_output  = torch.unsqueeze(validation_sample['output_data'], 0).to(device)
            result = model(val_d_in, attention_mask=val_d_attn, decoder_input_ids=val_d_decin, labels=val_d_output)

            val_loss = result["loss"]

            targetSec = val_d_output[0]

            oneAnswer = torch.argmax(result["logits"][0], dim=1)
            t = targetSec.size(0)

            t = targetSec[targetSec!=-100].size(0)
            c = (oneAnswer[:t] == targetSec[targetSec!=-100]).sum().item()
            w = (oneAnswer[:t] != targetSec[targetSec!=-100]).sum().item()
            # c = (oneAnswer == targetSec).sum().item()
            # w = (oneAnswer != targetSec).sum().item()

            acc = c/t

            if (len(rolling_val_acc) >= 20):
                rolling_val_acc.pop(0)

            if (len(rolling_val_loss) >= 20):
                rolling_val_loss.pop(0)

            rolling_val_acc.append(acc)
            rolling_val_loss.append(val_loss.item())

            run.log({"val_loss": val_loss.item(), "val_accuracy": acc, "val_loss_20rolling": statistics.mean(rolling_val_loss), "val_accuracy_20rolling": statistics.mean(rolling_val_acc)})

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

        oneAnswer = torch.argmax(logits[0], dim=1)
        answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)

        targetSec = output_data[0]

        t = targetSec[targetSec!=-100].size(0)
        c = (oneAnswer[:t] == targetSec[targetSec!=-100]).sum().item()
        w = (oneAnswer[:t] != targetSec[targetSec!=-100]).sum().item()

        acc = c/t
        avg_acc = (avg_acc+acc)/2
        max_acc = max(max_acc, acc)

        try: 
            answer = tokenizer.convert_tokens_to_string([a for a in answer_tokens[1:answer_tokens.index("</s>")] if a != tokenizer.pad_token])
        except ValueError:
            answer = tokenizer.convert_tokens_to_string([a for a in answer_tokens[1:] if a != tokenizer.pad_token])

        desiredAnswer_tokens = tokenizer.convert_ids_to_tokens(decoder_data[0])
        desiredAnswer = tokenizer.convert_tokens_to_string([a for a in desiredAnswer_tokens[2:] if a != tokenizer.pad_token])

        inputWord_tokens = tokenizer.convert_ids_to_tokens(input_data[0])
        inputWord = tokenizer.convert_tokens_to_string([a for a in inputWord_tokens if a != tokenizer.pad_token])

        if (i % 10 == 0):
            run.log({"loss": loss.item(),
                     "accuracy": acc,
                     "input": wandb.Html(inputWord),
                     "logits": wandb.Histogram(logits[0].detach().cpu()),
                     "output": wandb.Html(answer),
                     "target": wandb.Html(desiredAnswer)
                   })

            run.summary["max_accuracy"] = max_acc
            run.summary["avg_accuracy"] = avg_acc

#         writer.add_text('Train/sample', 
                # "<logits>"+answer+"</logits>\n\n"+
                # "<labels>"+desiredAnswer+"</labels>\n\n"+
                # "<src>"+inputWord+"</src>\n",
            # i+(epoch*len(databatched_loader)))

    # model.save_pretrained(f"./training/bart_enwiki_{epoch}-{modelID}")


