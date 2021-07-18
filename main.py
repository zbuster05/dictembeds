# type: ignore
# pylint: disable=no-member

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import statistics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
import re
import random
import uuid
import tqdm
import wandb
import json
import os

sys.setrecursionlimit(200000) 

hyperparametre_defaults = dict(
        learning_rate = 8e-5,
        num_warmup_steps = 4500,
        batch_size = 4,
        max_length = 250,
        base_model = 'facebook/bart-base',
        oc_mix = 0.1,
        val_mix = 0.1,
        noise_mix = 0.1,
        wiki = 'enwiki',
        max_steps = 50000
    )

# run = wandb.init(project='dictembed', entity='inscriptio', config=hyperparametre_defaults, mode="disabled")
run = wandb.init(project='dictembed', entity='inscriptio', config=hyperparametre_defaults)
config = wandb.config

training_data_originals = []

print("Caching originals data...")
for i in tqdm.tqdm(range(0,8)):
    filename = f"./data/{config.wiki}-parsed-oc-MD{i}.json"
    with open(filename, "r") as df:
        training_data_originals = training_data_originals + json.load(df)

validation_count = int(len(training_data_originals)*config.val_mix)
validation_data_originals = training_data_originals[:validation_count]
training_data_originals = training_data_originals[validation_count:]

training_data_oc = []
print("Caching OC data...")
for i in tqdm.tqdm(range(0,8)):
    filename = f"./data/{config.wiki}-parsed-oc-OC{i}.json"
    with open(filename, "r") as df:
        training_data_oc = training_data_oc + json.load(df)

oc_count = int(min(len(training_data_oc), (len(training_data_originals)//(1-config.oc_mix))*config.oc_mix))
oc_val_count = int(oc_count*config.val_mix)
validation_data_oc = training_data_oc[:oc_val_count]
training_data_oc = training_data_oc[oc_val_count:oc_val_count+oc_count]

training_data = training_data_originals+training_data_oc
validation_data = validation_data_originals+validation_data_oc

tokenizer = BartTokenizer.from_pretrained(config.base_model)

# https://stackoverflow.com/questions/46444656/bleu-scores-could-i-use-nltk-translate-bleu-score-sentence-bleu-for-calculating
smoothie = SmoothingFunction().method4

class EnWikiKeywordSentsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __getitem__(self, idx):
        is_noise = random.uniform(0,1)<config.noise_mix
        noise_index = random.randint(idx, len(self)-1)

        tokenizer = self.tokenizer
        max_length = self.max_length

        input_string = self.data[noise_index if is_noise else idx]["context"] 
        title_string = self.data[idx]["title"].lower()
        output_string = "<CND>" if is_noise else re.sub("(&.*?;)", "", re.sub("[{|}]", "", self.data[idx]["target"]))

        try: 
            if output_string[-1] not in ['.', '?', '>', '!', '"']:
                return self.__getitem__(random.randint(0, idx))
        except IndexError:
            return self.__getitem__(random.randint(0, idx))
 
        if len(self.data[idx]["target"]) < 45:
            return self.__getitem__(random.randint(0, idx))

        title_tokenized = tokenizer.tokenize(title_string)
        input_tokenized = [tokenizer.bos_token] + title_tokenized + [tokenizer.mask_token] + tokenizer.tokenize(input_string) + [tokenizer.eos_token]
        output_tokenized = tokenizer.encode(output_string)

        if len(output_tokenized) > max_length or len(input_tokenized) > max_length:
            return self.__getitem__(random.randint(0, idx))

        input_padded = input_tokenized + [tokenizer.pad_token for _ in range(max_length-len(input_tokenized))]

        input_encoded = tokenizer.convert_tokens_to_ids(input_padded)
        output_encoded = output_tokenized + [-100 for _ in range(max_length-len(output_tokenized))]

        input_mask = [1 for _ in range(len(input_tokenized))] + [0 for _ in range(max_length-len(input_tokenized))]

        if len(input_encoded) > max_length:
            return self.__getitem__(random.randint(0, idx))

        return {"input_data": torch.LongTensor(input_encoded), "output_data": torch.LongTensor(output_encoded), "input_mask": torch.LongTensor(input_mask)}

    def __len__(self):
        return len(self.data)-1

bart_config = BartConfig.from_pretrained(config.base_model)
# bart_config.output_past = True # https://github.com/huggingface/transformers/issues/3527
# bart_config.task_specific_params["summarization"]["max_length"] = config.max_length
# bart_config.task_specific_params["summarization_cnn"]["max_length"] = config.max_length
model = BartForConditionalGeneration.from_pretrained(config.base_model, config=bart_config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

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
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = config.num_warmup_steps, num_training_steps = config.max_steps)

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

max_bleu = 0
avg_bleu = 0

rolling_val_acc = []
rolling_val_loss = []
rolling_val_bleu = []

epochs = 0
steps = 0

min_val_20rolling = 1000

while steps < config.max_steps:
    databatched_loader = tqdm.tqdm(train_loader)

    # writer = SummaryWriter(f'./training/{modelID}')
    for i, chicken in enumerate(databatched_loader):
        # if (i % 10000 == 0 and i != 0):
        if (i % 10000 == 0 and i != 0):
            # artifact = wandb.Artifact(f'bart_{config.wiki}-kw_summary', type='model', description="BART model finetuned upon enwiki first sentences")
            tokenizer.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:ROUTINE::{epochs}:{i}")
            model.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:ROUTINE::{epochs}:{i}")
            # artifact.add_dir("./training/bart_{config.wiki}-kw_summary-{modelID}:{epoch}:{i}")
            # run.log_artifact(artifact)

        if (i % 100 == 0):
            validation_index = random.randint(0, len(validate_dataset))
            validation_sample = validate_dataset[validation_index]
            val_d_in = torch.unsqueeze(validation_sample['input_data'], 0).to(device)
            val_d_attn  = torch.unsqueeze(validation_sample['input_mask'], 0).to(device)
            val_d_output  = torch.unsqueeze(validation_sample['output_data'], 0).to(device)
            result = model(val_d_in, attention_mask=val_d_attn, labels=val_d_output)

            val_loss = result["loss"]

            targetSec = val_d_output[0]

            oneAnswer = torch.argmax(result["logits"][0], dim=1)
            answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)

            try: 
                answer_tokens_clear = [a for a in answer_tokens[0:answer_tokens.index("</s>")+1] if a != tokenizer.pad_token]
            except ValueError:
                answer_tokens_clear = [a for a in answer_tokens[0:] if a != tokenizer.pad_token]

            answer = tokenizer.convert_tokens_to_string(answer_tokens_clear)

            desiredAnswer_tokens = list(filter(lambda x:x, tokenizer.convert_ids_to_tokens(targetSec)))
            desiredAnswer = tokenizer.convert_tokens_to_string(desiredAnswer_tokens)

            t = targetSec.size(0)

            t = targetSec[targetSec!=-100].size(0)
            c = (oneAnswer[:t] == targetSec[targetSec!=-100]).sum().item()
            w = (oneAnswer[:t] != targetSec[targetSec!=-100]).sum().item()
            # c = (oneAnswer == targetSec).sum().item()
            # w = (oneAnswer != targetSec).sum().item()

            acc = c/t
            try: 
                bleu = sentence_bleu([desiredAnswer_tokens], answer_tokens_clear, smoothing_function=smoothie)
            except ValueError:
                continue

            if "<CND>" not in desiredAnswer:
                if (len(rolling_val_acc) >= 20):
                    rolling_val_acc.pop(0)

                if (len(rolling_val_loss) >= 20):
                    rolling_val_loss.pop(0)

                if (len(rolling_val_bleu) >= 20):
                    rolling_val_bleu.pop(0)

            rolling_val_acc.append(acc)
            rolling_val_loss.append(val_loss.item())
            rolling_val_bleu.append(bleu)

             # if we have a new min                                  # if we haden't just started
            if statistics.mean(rolling_val_loss)<(min_val_20rolling-0.1) and i > 10000:
                min_val_20rolling = statistics.mean(rolling_val_loss)

                # saving "best" weights
                tokenizer.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:B_VAL::{epochs}:{i}:{min_val_20rolling}")
                model.save_pretrained(f"./training/bart_{config.wiki}-kw_summary-{modelID}:B_VAL::{epochs}:{i}:{min_val_20rolling}")

                

            run.log({"val_loss": val_loss.item(), "val_accuracy": acc, "val_bleu": bleu, "val_loss_20rolling": statistics.mean(rolling_val_loss), "val_accuracy_20rolling": statistics.mean(rolling_val_acc), "val_bleu_20rolling": statistics.mean(rolling_val_bleu)})

        optim.zero_grad()

        input_data = chicken['input_data'].to(device)
        output_data = chicken['output_data'].to(device)
        attention_mask = chicken['input_mask'].to(device)

        result = model(input_data, attention_mask=attention_mask, labels=output_data)
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
            answer_tokens_clear = [a for a in answer_tokens[0:answer_tokens.index("</s>")+1] if a != tokenizer.pad_token]
        except ValueError:
            answer_tokens_clear = [a for a in answer_tokens[0:] if a != tokenizer.pad_token]

        answer = tokenizer.convert_tokens_to_string(answer_tokens_clear)

        desiredAnswer_tokens = list(filter(lambda x:x, tokenizer.convert_ids_to_tokens(targetSec)))
        desiredAnswer = tokenizer.convert_tokens_to_string(desiredAnswer_tokens)

        inputWord_tokens = [a for a in tokenizer.convert_ids_to_tokens(input_data[0]) if a != tokenizer.pad_token]
        inputWord = tokenizer.convert_tokens_to_string(inputWord_tokens)

        try: 
            bleu = sentence_bleu([desiredAnswer_tokens], answer_tokens_clear, smoothing_function=smoothie)
        except ValueError:
            continue

        avg_bleu = (avg_bleu+bleu)/2
        max_bleu = max(max_bleu, bleu)

        if (i % 10 == 0):
            try: 
                run.log({"loss": loss.item(),
                         "accuracy": acc,
                         "bleu": bleu,
                         "input": wandb.Html(inputWord[3:-4]),
                         "logits": wandb.Histogram(logits[0].detach().cpu()),
                         "output": wandb.Html(answer[3:-4]),
                         "target": wandb.Html(desiredAnswer[3:-4])
                       })

                run.summary["max_accuracy"] = max_acc
                run.summary["avg_accuracy"] = avg_acc

                run.summary["max_bleu"] = max_bleu
                run.summary["avg_bleu"] = avg_bleu
                
                run.summary["epochs"] = epochs

            except IsADirectoryError:
                print("um.")

        steps += 1

        if steps >= config.max_steps:
            break

    epochs += 1
#         writer.add_text('Train/sample', 
                # "<logits>"+answer+"</logits>\n\n"+
                # "<labels>"+desiredAnswer+"</labels>\n\n"+
                # "<src>"+inputWord+"</src>\n",
            # i+(epoch*len(databatched_loader)))

    # model.save_pretrained(f"./training/bart_enwiki_{epoch}-{modelID}")


