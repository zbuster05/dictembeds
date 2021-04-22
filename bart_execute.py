# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

import uuid
import tqdm
import json
import os


tokenizer = BartTokenizer.from_pretrained("./training/bart_enwiki_BASE-05fcd:0:1000")

model = BartForConditionalGeneration.from_pretrained("./training/bart_enwiki_BASE-05fcd:0:1000")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)


ARTICLE_TO_SUMMARIZE = """
Whole segments of genome exchange abruptly as two flu viruses infect the same cell to create a new strand.  There are two mechnisms by which happens — the crossing-over mechnism and genome segment reassortment Self-mixing of ozaki fragments during viral recombination in the ozaki process cause sudden mutations. Two viruses coinfect the same cell, causing cross-talk in swapping segment.

This usually occurs due an error in a polymerase-driven process, where single/groups of nucleotides flip slowly over time due to mistakes in replication. The former is an environment-dependent process, where the latter is able to be modeled as it is due to predictable transcription mistake.
"""
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=512, return_tensors='pt')

summary_ids = model.generate(inputs['input_ids'], num_beams=4, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
breakpoint()

# model.resize_token_embeddings(len(tokenizer))

# model.train()

# train_dataset = EnWikiKeywordSentsDataset(tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# optim = AdamW([
    # {"params": model.transformer.h.parameters(), "lr": 5e-6},
        # {"params": model.lm_head.parameters(), "lr": 1e-5},
    # ], lr=1e-5)

# optim = AdamW(model.parameters(), lr=3e-5)
# scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = 1000, num_training_steps = 3*len(train_loader))

# modelID = str(uuid.uuid4())[-5:]

# model.save_pretrained(f"./training/bart_enwiki_BASE-{modelID}")
# # https://huggingface.co/transformers/custom_datasets.html?highlight=fine%20tuning
# # model.resize_token_embeddings(len(tokenizer))
# for epoch in range(3):
    # databatched_loader = tqdm.tqdm(train_loader)

    # writer = SummaryWriter(f'./training/{modelID}')
    # for i, chicken in enumerate(databatched_loader):
        # optim.zero_grad()

        # input_data = chicken['input_data'].to(device)
        # output_data = chicken['output_data'].to(device)
        # attention_mask = chicken['input_mask'].to(device)

        # result = model(input_data, attention_mask=attention_mask, labels=output_data)
        # logits = result["logits"]
        # loss = result["loss"]

        # databatched_loader.set_description(f'{modelID} loss: {loss}')
        # databatched_loader.refresh()
    
        # loss.backward()
        # optim.step()
        # scheduler.step()

        # oneAnswer = torch.argmax(logits[0], dim=1)
        # answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)
        # answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        # writer.add_scalar('Train/loss', loss.item(), i+(epoch*len(databatched_loader)))
        # writer.add_text('Train/sample', answer, i+(epoch*len(databatched_loader)))
    
    # model.save_pretrained(f"./training/bart_enwiki_{epoch}-{modelID}")


