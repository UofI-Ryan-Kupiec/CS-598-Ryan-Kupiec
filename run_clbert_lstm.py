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
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import BertTokenizer, BertConfig
from torch.optim import AdamW
from modeling_readmission import BertModel as ClinicalBertModel
from modeling_patient import LSTMLayer
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from dotmap import DotMap
import re
from other_func import write_log, Tokenize_with_note_id, concat_by_id_list_with_note_chunk_id, convert_note_ids, \
    flat_accuracy, write_performance, reorder_by_time
from utils import time_batch_generator


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--train_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input training data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--val_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input validation data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--test_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--log_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The log file path.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--save_model",
                        default=False,
                        action='store_true',
                        help="Whether to save the model.")

    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--embed_mode",
                        default=None,
                        type=str,
                        required=True,
                        help="The embedding type selected in the list: all, note, chunk, no.")

    parser.add_argument("--task_name",
                        default="LSTM_with_ClBERT_mortality",
                        type=str,
                        required=True,
                        help="The name of the task.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_chunk_num",
                        default=64,
                        type=int,
                        help="The maximum total input chunk numbers after WordPiece tokenization.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ——————————————————————————————————————————————————————————
    # 1) set up logging, config, device
    # ——————————————————————————————————————————————————————————
    LOG = args.log_path
    LOG_PATH = args.log_path
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

    write_log("Tokenize Start!", LOG_PATH)
    tl, ti, tm, tni= Tokenize_with_note_id(train_df, MAX_LEN, tokenizer)
    vl, vi, vm, vni= Tokenize_with_note_id(val_df,   MAX_LEN, tokenizer)
    el, ei, em, eni= Tokenize_with_note_id(test_df,  MAX_LEN, tokenizer)
    write_log("Tokenize Finished!", LOG_PATH)
    
    # convert to torch tensors
    # new: explicitly cast times to float32
    train_inputs, train_masks, train_labels = (
        torch.tensor(ti, dtype=torch.long),
        torch.tensor(tm, dtype=torch.long),
        torch.tensor(tl, dtype=torch.float)
    )
    val_inputs, val_masks, val_labels = (
        torch.tensor(vi, dtype=torch.long),
        torch.tensor(vm, dtype=torch.long),
        torch.tensor(vl, dtype=torch.float)
    )
    test_inputs, test_masks, test_labels = (
        torch.tensor(ei, dtype=torch.long),
        torch.tensor(em, dtype=torch.long),
        torch.tensor(el, dtype=torch.float)
    )
    
    # group by admission
    train_labels, train_inputs, train_masks, train_ids, train_note_ids, train_chunk_ids = \
      concat_by_id_list_with_note_chunk_id(train_df, train_labels, train_inputs, train_masks,
                                                tni, MAX_LEN)

    validation_labels, validation_inputs, validation_masks, validation_ids, validation_note_ids, validation_chunk_ids = \
      concat_by_id_list_with_note_chunk_id(val_df, vl, val_inputs, val_masks,
                                                vni, MAX_LEN)

    test_labels, test_inputs, test_masks, test_ids, test_note_ids, test_chunk_ids = \
      concat_by_id_list_with_note_chunk_id(test_df, el, test_inputs, test_masks,
                                                eni, MAX_LEN)

    # ——————————————————————————————————————————————————————————
    # 3) build models
    # ——————————————————————————————————————————————————————————
    lstm = LSTMLayer(config=config, num_labels=1).to(device)

    if n_gpu > 1:
        bert = nn.DataParallel(bert)
        lstm = nn.DataParallel(lstm)
    
    # ——————————————————————————————————————————————————————————
    # 4) optimizer + scheduler setup
    # ——————————————————————————————————————————————————————————
    no_decay = ['bias', 'gamma', 'beta']
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
    start = time.time()
    train_loss_history = []
    gen = time_batch_generator(
        args.max_chunk_num,
        train_inputs, train_labels, train_masks,
        train_note_ids, train_chunk_ids
    )
    gen_val = time_batch_generator(
            args.max_chunk_num,
            validation_inputs, validation_labels, validation_masks,
            validation_note_ids, validation_chunk_ids
        )

    write_log("Training start!", LOG_PATH)
    for epoch in range(int(args.num_train_epochs)):
        bert.train(); lstm.train()
        tr_loss = 0.0
        for step in range(len(train_ids)):
            b_input_ids, b_labels, b_input_mask, b_note_ids, b_chunk_ids = next(gen)
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_new_note_ids = convert_note_ids(b_note_ids).to(device)
            b_chunk_ids = b_chunk_ids.unsqueeze(0).to(device)
            b_labels = b_labels.to(device).unsqueeze(0)
            optimizer.zero_grad()
            _, whole_output = bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            whole_input = whole_output.unsqueeze(0)
            loss, pred = lstm(whole_input, b_new_note_ids.unsqueeze(0), b_chunk_ids, b_labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
        avg_train_loss = tr_loss / len(train_ids)
        write_log(f"[Epoch {epoch}] train loss = {avg_train_loss:.4f}", LOG)
        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        bert.eval(); lstm.eval()
        val_acc = 0.0
        val_steps = 0
        for _ in range(len(validation_ids)):
            with torch.no_grad():
                b_input_ids, b_labels, b_input_mask, b_note_ids, b_chunk_ids = next(gen_val)
                b_input_ids = b_input_ids.to(device)
                b_input_mask = b_input_mask.to(device)
                b_new_note_ids = convert_note_ids(b_note_ids).to(device)
                b_chunk_ids = b_chunk_ids.unsqueeze(0).to(device)
                b_labels = torch.tensor(b_labels, dtype=torch.float, device=device).unsqueeze(0)
                _, whole_output = bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                whole_input = whole_output.unsqueeze(0)
                pred = lstm(whole_input, b_new_note_ids.unsqueeze(0), b_chunk_ids)
            val_acc += flat_accuracy(pred.detach().cpu().numpy(),
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
    
    # ——————————————————————————————————————————————————————————
    # 7) run on test set and save predictions
    # ——————————————————————————————————————————————————————————
    bert.eval(); lstm.eval()
    tst_acc = 0.0
    tst_steps = 0
    gen_test = time_batch_generator(
        args.max_chunk_num, test_inputs, test_labels, test_masks,
            test_note_ids, test_chunk_ids
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

if __name__ == "__main__":
    main()
