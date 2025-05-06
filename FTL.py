#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

import time
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ←—— updated imports —————————————————————————————————————————————
from transformers import BertTokenizer, BertConfig
# ——————————————————————————————————————————————————————————————————————

from modeling_readmission import BertModel as ClinicalBertModel
from modeling_patient import FTLSTMLayer
from other_func import (
    write_log,
    Tokenize_with_note_id_hour,
    concat_by_id_list_with_note_chunk_id_time,
    convert_note_ids,
    flat_accuracy,
    write_performance,
    reorder_by_time,
)
from utils import time_batch_generator
from dotmap import DotMap
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--train_data",    type=str, required=True)
    parser.add_argument("--val_data",      type=str, required=True)
    parser.add_argument("--test_data",     type=str, required=True)
    parser.add_argument("--log_path",      type=str, required=True)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--save_model",    action="store_true")
    parser.add_argument("--bert_model",    type=str, required=True)
    parser.add_argument("--embed_mode",    type=str, required=True)
    parser.add_argument("--task_name",     type=str, required=True)
    ## Other parameters
    parser.add_argument("--max_seq_length",        type=int,   default=128)
    parser.add_argument("--max_chunk_num",         type=int,   default=64)
    parser.add_argument("--train_batch_size",      type=int,   default=1)
    parser.add_argument("--eval_batch_size",       type=int,   default=1)
    parser.add_argument("--learning_rate",         type=float, default=2e-5)
    parser.add_argument("--warmup_proportion",     type=float, default=0.1)
    parser.add_argument("--num_train_epochs",      type=int,   default=3)
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ——————————————————————————————————————————————————————————
    # 1) set up logging, config, device
    # ——————————————————————————————————————————————————————————
    LOG = args.log_path
    MAX_LEN = args.max_seq_length
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    bert_cfg = BertConfig.from_pretrained(args.bert_model)
    bert = ClinicalBertModel.from_pretrained(args.bert_model).to(device)
    
    config = DotMap(
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        max_note_position_embedding=1000,
        max_chunk_position_embedding=1000,
        embed_mode=args.embed_mode,
        hidden_size = bert_cfg.hidden_size,
        lstm_layers=1,
        task_name=args.task_name,
    )

    write_log(f"New Job Start! Config: {config.toDict()}", LOG)

    # ——————————————————————————————————————————————————————————
    # 2) load & tokenize data
    # ——————————————————————————————————————————————————————————
    def load_df(fn):
        return pd.read_csv(os.path.join(args.data_dir, fn))

    train_df = reorder_by_time(load_df(args.train_data))
    val_df   = reorder_by_time(load_df(args.val_data))
    test_df  = reorder_by_time(load_df(args.test_data))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    write_log("Tokenizing...", LOG)
    tl, ti, tm, tni, tt = Tokenize_with_note_id_hour(train_df, MAX_LEN, tokenizer)
    vl, vi, vm, vni, vt = Tokenize_with_note_id_hour(val_df,   MAX_LEN, tokenizer)
    el, ei, em, eni, et = Tokenize_with_note_id_hour(test_df,  MAX_LEN, tokenizer)
    write_log("Tokenization done.", LOG)

    # convert to torch tensors
    # new: explicitly cast times to float32
    train_inputs, train_masks, train_labels, train_times = (
        torch.tensor(ti, dtype=torch.long),
        torch.tensor(tm, dtype=torch.long),
        torch.tensor(tl, dtype=torch.float),
        torch.tensor(tt, dtype=torch.float),
    )
    val_inputs, val_masks, val_labels, val_times = (
        torch.tensor(vi, dtype=torch.long),
        torch.tensor(vm, dtype=torch.long),
        torch.tensor(vl, dtype=torch.float),
        torch.tensor(vt, dtype=torch.float),
    )
    test_inputs, test_masks, test_labels, test_times = (
        torch.tensor(ei, dtype=torch.long),
        torch.tensor(em, dtype=torch.long),
        torch.tensor(el, dtype=torch.float),
        torch.tensor(et, dtype=torch.float),
    )


    # group by admission
    train_labels, train_inputs, train_masks, train_ids, train_note_ids, train_chunk_ids, train_times = \
      concat_by_id_list_with_note_chunk_id_time(train_df, train_labels, train_inputs, train_masks,
                                                tni, train_times, MAX_LEN)

    validation_labels, validation_inputs, validation_masks, validation_ids, validation_note_ids, validation_chunk_ids, validation_times = \
      concat_by_id_list_with_note_chunk_id_time(val_df, vl, val_inputs, val_masks,
                                                vni, val_times, MAX_LEN)

    test_labels, test_inputs, test_masks, test_ids, test_note_ids, test_chunk_ids, test_times = \
      concat_by_id_list_with_note_chunk_id_time(test_df, el, test_inputs, test_masks,
                                                eni, test_times, MAX_LEN)

    # ——————————————————————————————————————————————————————————
    # 3) build models
    # ——————————————————————————————————————————————————————————
    lstm = FTLSTMLayer(config=config, num_labels=1).to(device)

    if n_gpu > 1:
        bert = nn.DataParallel(bert)
        lstm = nn.DataParallel(lstm)

    # ——————————————————————————————————————————————————————————
    # 4) optimizer + scheduler setup
    # ——————————————————————————————————————————————————————————
    # gather all parameters
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped = [
        {
            "params": [p for n, p in list(bert.named_parameters()) + list(lstm.named_parameters())
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in list(bert.named_parameters()) + list(lstm.named_parameters())
                       if     any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped, lr=args.learning_rate, weight_decay=0.01)

    # total steps = #batches * epochs / gradient_accumulation_steps
    t_total = (
        len(train_ids)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    warmup_steps = int(args.warmup_proportion * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = warmup_steps,
        num_training_steps = t_total
    )

    # ——————————————————————————————————————————————————————————
    # 5) training loop
    # ——————————————————————————————————————————————————————————
    train_loss_history = []
    gen = time_batch_generator(
        args.max_chunk_num,
        train_inputs, train_labels, train_masks,
        train_note_ids, train_chunk_ids, train_times,
    )

    for epoch in range(int(args.num_train_epochs)):
        bert.train(); lstm.train()
        tr_loss = 0.0

        for step in range(len(train_ids)):
            b_input_ids, b_labels, b_input_mask, b_note_ids, b_chunk_ids, b_times = next(gen)
            b_input_ids = b_input_ids.to(device)
            b_input_mask= b_input_mask.to(device)
            b_new_note_ids = convert_note_ids(b_note_ids).to(device)
            b_chunk_ids = b_chunk_ids.unsqueeze(0).to(device)
            b_times     = b_times.unsqueeze(0).to(device)
            #b_labels    = b_labels.to(device)
            b_labels = b_labels.to(device).unsqueeze(0)

            optimizer.zero_grad()

            # 1) encode via BERT
            _, pooled_output = bert(
                b_input_ids,
                attention_mask=b_input_mask,
                token_type_ids=None,
            )
            pooled_output = pooled_output.unsqueeze(0)  # [1, chunks, hidden]

            # 2) time-aware LSTM
            loss, logits = lstm(
                pooled_output,
                b_times,
                b_new_note_ids.unsqueeze(0),
                b_chunk_ids,
                b_labels,
            )

            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()

        avg_train_loss = tr_loss / len(train_ids)
        write_log(f"[Epoch {epoch}] train loss = {avg_train_loss:.4f}", LOG)

        # — validation pass (optional) —
        bert.eval(); lstm.eval()
        val_acc = 0.0
        val_steps = 0
        gen_val = time_batch_generator(
            args.max_chunk_num,
            validation_inputs, validation_labels, validation_masks,
            validation_note_ids, validation_chunk_ids, validation_times,
        )
        for _ in range(len(validation_ids)):
            with torch.no_grad():
                b_input_ids, b_labels, b_input_mask, b_note_ids, b_chunk_ids, b_times = next(gen_val)
                b_input_ids = b_input_ids.to(device)
                b_input_mask= b_input_mask.to(device)
                b_new_note_ids = convert_note_ids(b_note_ids).to(device)
                b_chunk_ids = b_chunk_ids.unsqueeze(0).to(device)
                b_times     = b_times.unsqueeze(0).to(device)
                b_labels = torch.tensor(b_labels, dtype=torch.float, device=device).unsqueeze(0)
                #b_labels = b_labels.resize_(1) 

                _, pooled_output = bert(
                    b_input_ids, attention_mask=b_input_mask, token_type_ids=None
                )
                pooled_output = pooled_output.unsqueeze(0)
                preds = lstm(
                    pooled_output, b_times, b_new_note_ids.unsqueeze(0), b_chunk_ids
                )  # second return = logits

            val_acc += flat_accuracy(preds.detach().cpu().numpy(),
                                     b_labels.detach().cpu().numpy())
            val_steps += 1

        write_log(f"[Epoch {epoch}] val acc = {val_acc/val_steps:.4f}", LOG)
        
        # optionally save a checkpoint…
        if args.save_model:
            ckpt = {
                "epoch": epoch,
                "bert_state": bert.module.state_dict() if n_gpu>1 else bert.state_dict(),
                "lstm_state": lstm.module.state_dict() if n_gpu>1 else lstm.state_dict(),
                "optim_state": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pt"))

    fig1 = plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_loss_history, marker='o')
    plt.grid(True)

    loss_fig_path = os.path.join(args.output_dir, "training_loss.png")
    plt.savefig(loss_fig_path, dpi=fig1.dpi)
    write_log(f"Saved training loss plot to {loss_fig_path}", LOG)

    # ——————————————————————————————————————————————————————————
    # 7) run on test set and save predictions
    # ——————————————————————————————————————————————————————————
    bert.eval()
    lstm.eval()
    tst_acc = 0.0
    tst_steps = 0
    gen_val = time_batch_generator(
        args.max_chunk_num, test_inputs, test_labels, test_masks,
            test_note_ids, test_chunk_ids, test_times
    )
    for _ in range(len(test_ids)):
        with torch.no_grad():
            b_input_ids, b_labels, b_input_mask, b_note_ids, b_chunk_ids, b_times = next(gen_val)
            b_input_ids = b_input_ids.to(device)
            b_input_mask= b_input_mask.to(device)
            b_new_note_ids = convert_note_ids(b_note_ids).to(device)
            b_chunk_ids = b_chunk_ids.unsqueeze(0).to(device)
            b_times     = b_times.unsqueeze(0).to(device)
            b_labels = torch.tensor(b_labels, dtype=torch.float, device=device).unsqueeze(0)

            _, pooled_output = bert(
                b_input_ids, attention_mask=b_input_mask, token_type_ids=None
            )
            pooled_output = pooled_output.unsqueeze(0)
            preds = lstm(
                pooled_output, b_times, b_new_note_ids.unsqueeze(0), b_chunk_ids
            )  # second return = logits

        tst_acc += flat_accuracy(preds.detach().cpu().numpy(),
                                b_labels.detach().cpu().numpy())
        tst_steps += 1
    write_log(f"[Epoch {epoch}] test acc = {tst_acc/tst_steps:.4f}", LOG)
    
    write_log("Training complete!", LOG)

if __name__ == "__main__":
    main()