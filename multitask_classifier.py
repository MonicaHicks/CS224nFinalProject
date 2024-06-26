'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
  copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
  the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

import gc

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from tokenizer import BertTokenizer

TQDM_DISABLE = False

pos_tags = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
    'WRB'
]
pos2idx = {tag: idx for idx, tag in enumerate(pos_tags)}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT parameters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        # TODO
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sst_linear = nn.Linear(self.hidden_size, N_SENTIMENT_CLASSES)
        self.para_linear = nn.Linear(self.hidden_size + self.hidden_size, 1)
        self.sts_linear = nn.Linear(self.hidden_size + self.hidden_size, 1)

        # extension added
        # https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/
        self.pos_tags_embedding_dim = config.hidden_size
        self.pos_tags_embedding = nn.Embedding(len(pos2idx), self.pos_tags_embedding_dim)
        self.projection = nn.Linear(config.hidden_size + self.pos_tags_embedding_dim, config.hidden_size)
        # raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # TODO
        gc.collect()
        torch.cuda.empty_cache()
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output['pooler_output']
        # output = self.dropout(output)
        return output
        # raise NotImplementedError

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        sent = self.forward(input_ids, attention_mask)
        pos_tags_list = []
        for id_list in input_ids.tolist():
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            pos_tags = pos_tag(tokens)
            pos_tags_list.append(pos_tags)
    
        pos_tag_ids = [[pos2idx.get(tag, 0) for word, tag in sent_pos_tags] for sent_pos_tags in pos_tags_list]
        pos_tag_ids_tensor = torch.tensor(pos_tag_ids, dtype=torch.long, device=input_ids.device)
        pos_tags_embeds = self.pos_tags_embedding(pos_tag_ids_tensor)
        pos_tags_embeds = torch.mean(pos_tags_embeds, 1)
        concatenated_embeddings = torch.cat((sent, pos_tags_embeds), dim=-1)
        output = self.projection(concatenated_embeddings)
        return self.sst_linear(output)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        # TODO
        sent_1 = self.forward(input_ids_1, attention_mask_1)
        sent_2 = self.forward(input_ids_2, attention_mask_2)
        pos_tags_list_1 = []
        for id_list in input_ids_1.tolist():
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            pos_tags = pos_tag(tokens)
            pos_tags_list_1.append(pos_tags)
        pos_tags_list_2 = []
        for id_list in input_ids_2.tolist():
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            pos_tags = pos_tag(tokens)
            pos_tags_list_2.append(pos_tags)
    
        pos_tag_ids_1 = [[pos2idx.get(tag, 0) for word, tag in sent_pos_tags] for sent_pos_tags in pos_tags_list_1]
        pos_tag_ids_tensor_1 = torch.tensor(pos_tag_ids_1, dtype=torch.long, device=input_ids_1.device)
        pos_tags_embeds_1 = self.pos_tags_embedding(pos_tag_ids_tensor_1)
        pos_tags_embeds_1 = torch.mean(pos_tags_embeds_1, 1)
        concatenated_embeddings_1 = torch.cat((sent_1, pos_tags_embeds_1), dim=-1)
        output_1 = self.projection(concatenated_embeddings_1)

        pos_tag_ids_2 = [[pos2idx.get(tag, 0) for word, tag in sent_pos_tags] for sent_pos_tags in pos_tags_list_2]
        pos_tag_ids_tensor_2 = torch.tensor(pos_tag_ids_2, dtype=torch.long, device=input_ids_2.device)
        pos_tags_embeds_2 = self.pos_tags_embedding(pos_tag_ids_tensor_2)
        pos_tags_embeds_2 = torch.mean(pos_tags_embeds_2, 1)
        concatenated_embeddings_2 = torch.cat((sent_2, pos_tags_embeds_2), dim=-1)
        output_2 = self.projection(concatenated_embeddings_2)
        cat = torch.cat((output_1, output_2), 1)
        return self.para_linear(cat)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # sent_1 = self.forward(input_ids_1, attention_mask_1)
        # sent_2 = self.forward(input_ids_2, attention_mask_2)
        # cat = torch.cat((sent_1, sent_2), 1)
        # output = self.sts_linear(cat)
        # return output
        sent_1 = self.forward(input_ids_1, attention_mask_1)
        sent_2 = self.forward(input_ids_2, attention_mask_2)

        pos_tags_list_1 = []
        for id_list in input_ids_1.tolist():
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            pos_tags = pos_tag(tokens)
            pos_tags_list_1.append(pos_tags)
        pos_tags_list_2 = []
        for id_list in input_ids_2.tolist():
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            pos_tags = pos_tag(tokens)
            pos_tags_list_2.append(pos_tags)
    
        pos_tag_ids_1 = [[pos2idx.get(tag, 0) for word, tag in sent_pos_tags] for sent_pos_tags in pos_tags_list_1]
        pos_tag_ids_tensor_1 = torch.tensor(pos_tag_ids_1, dtype=torch.long, device=input_ids_1.device)
        pos_tags_embeds_1 = self.pos_tags_embedding(pos_tag_ids_tensor_1)
        pos_tags_embeds_1 = torch.mean(pos_tags_embeds_1, 1)
        concatenated_embeddings_1 = torch.cat((sent_1, pos_tags_embeds_1), dim=-1)
        output_1 = self.projection(concatenated_embeddings_1)

        pos_tag_ids_2 = [[pos2idx.get(tag, 0) for word, tag in sent_pos_tags] for sent_pos_tags in pos_tags_list_2]
        pos_tag_ids_tensor_2 = torch.tensor(pos_tag_ids_2, dtype=torch.long, device=input_ids_2.device)
        pos_tags_embeds_2 = self.pos_tags_embedding(pos_tag_ids_tensor_2)
        pos_tags_embeds_2 = torch.mean(pos_tags_embeds_2, 1)
        concatenated_embeddings_2 = torch.cat((sent_2, pos_tags_embeds_2), dim=-1)
        output_2 = self.projection(concatenated_embeddings_2)
        cat = torch.cat((output_1, output_2), 1)
        return self.sts_linear(cat)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn, num_workers=16)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn, num_workers=16)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        # train_loss = 0
        # num_batches = 0
        # SST
        #
        num_sst_batches = 0
        sst_loss = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # print(batch)
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            # print(logits.size(), b_labels.size(), b_labels.view(-1).size())
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            sst_loss += loss.item()
            num_sst_batches += 1
        print("sst loss is ", sst_loss, " batches is ", num_sst_batches)
        # '''
        
        # PARAPHRASE
        # '''
        num_para_batches = 0
        para_loss = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # print(batch)
            b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'], batch['token_ids_2'],
                                                              batch['attention_mask_1'], batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            lossfx = torch.nn.BCEWithLogitsLoss()
            loss = lossfx(logits[:, 0].to(torch.float32), b_labels.view(-1).to(torch.float32)) / args.batch_size

            loss.backward()
            optimizer.step()

            para_loss += loss.item()
            num_para_batches += 1
        #  '''
        print("para loss is ", para_loss, " batches is ", num_para_batches)
        # STS
        # '''
        num_sts_batches = 0
        sts_loss = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # print(batch)
            b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'], batch['token_ids_2'],
                                                              batch['attention_mask_1'], batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            # print(logits.size(), b_labels.size(), b_labels.view(-1).size())
            lossfx = torch.nn.MSELoss()
            # torch.set_grad_enabled(True)
            loss = lossfx(logits[:, 0].to(torch.float32), b_labels.view(-1).to(torch.float32)) / args.batch_size
            # loss.requires_grad = True 
            # print(loss)
            loss.backward()
            optimizer.step()

            sts_loss += loss.item()
            num_sts_batches += 1
        # '''
        print("sts loss is ", sts_loss, " batches is ", num_sts_batches)

        sst_loss /= num_sts_batches
        para_loss /= num_para_batches
        sts_loss /= num_sts_batches
        train_loss = (sst_loss + para_loss + sts_loss)

        # sst_train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # sst_dev_acc, dev_f1, *_  =  model_eval_sst(sst_dev_dataloader, model, device)

        # para_train_acc, train_f1, *_ = model_eval_sst(para_train_dataloader, model, device)
        # para_dev_acc, dev_f1, *_  =  model_eval_sst(para_dev_dataloader, model, device)

        # para_train_acc, train_f1, *_ = model_eval_sst(para_train_dataloader, model, device)
        # para_dev_acc, dev_f1, *_  =  model_eval_sst(para_dev_dataloader, model, device)

        train_acc = 0
        if epoch % 3 == 0:
            sentiment_train_accuracy, sst_y_pred, sst_sent_ids, \
            paraphrase_train_accuracy, para_y_pred, para_sent_ids, \
            sts_train_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_train_dataloader,
                                                                            para_train_dataloader,
                                                                            sts_train_dataloader, model, device)
            train_acc = (sentiment_train_accuracy + paraphrase_train_accuracy + sts_train_corr) / 3
        # print()
        sentiment_dev_accuracy, sst_y_pred, sst_sent_ids, \
        paraphrase_dev_accuracy, para_y_pred, para_sent_ids, \
        sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                      para_dev_dataloader,
                                                                      sts_dev_dataloader, model, device)
        dev_acc = (sentiment_dev_accuracy + paraphrase_dev_accuracy + sts_dev_corr) / 3

        if dev_acc > best_dev_acc:
            print("old best dev acc = ", best_dev_acc, ". saved model.")
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
        dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                              para_dev_dataloader,
                                                                              sts_dev_dataloader, model, device)

        test_sst_y_pred, \
        test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(sst_test_dataloader,
                                      para_test_dataloader,
                                      sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are 
updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
