import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_para, model_eval_sts, test_model_multitask

TQDM_DISABLE = False


# fix the random seed
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


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.af = nn.ReLU()
        self.conv1 = nn.Conv1d(2 * 768, 100, 3)
        self.conv2 = nn.Conv1d(2 * 768, 100, 4)
        self.conv3 = nn.Conv1d(2 * 768, 100, 5)

    def forward(self, embedding, constant_embedding):
        embedding = torch.cat((embedding, constant_embedding), dim=2)
        embedding = embedding.permute(0, 2, 1)
        x = torch.squeeze(self.af(self.pool(self.conv1(embedding))), dim=-1)

        y = torch.squeeze(self.af(self.pool(self.conv2(embedding))), dim=-1)

        z = torch.squeeze(self.af(self.pool(self.conv3(embedding))), dim=-1)

        output = torch.cat((x, y, z), dim=1)

        return output


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=config.local_files_only)
        ### TODO

        # Pretrain mode does not require updating bert parameters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # Sentiment classification
        self.TextCNN = TextCNN()
        self.sst_1 = nn.Linear(300, 100)
        self.sst_2 = nn.Linear(100, 5)
        self.sst_fc1 = nn.Linear(BERT_HIDDEN_SIZE, int(BERT_HIDDEN_SIZE / 2))
        self.sst_fc2 = nn.Linear(int(BERT_HIDDEN_SIZE / 2), 5)

        self.TextCNN_para = TextCNN()
        self.para_1 = nn.Linear(300 * 2, 300)
        self.para_2 = nn.Linear(300, 100)
        self.para_3 = nn.Linear(100, 1)
        self.para_fc1 = nn.Linear(BERT_HIDDEN_SIZE * 3, BERT_HIDDEN_SIZE * 2)
        self.para_fc2 = nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE)
        self.para_fc3 = nn.Linear(BERT_HIDDEN_SIZE, 1)

        self.similarity_1 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.similarity_2 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.similarity_3 = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.similarity_4 = nn.Linear(BERT_HIDDEN_SIZE, 1)
        self.similarity_5 = nn.Linear(2, 1)

        self.af = nn.ELU()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        return self.bert(input_ids, attention_mask)

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        x = self.forward(input_ids, attention_mask)
        constant_embedding = self.bert.embed(input_ids) * attention_mask.unsqueeze(-1)
        last_hidden_state = x['last_hidden_state'] * attention_mask.unsqueeze(-1)
        pooler_output = x['pooler_output']

        output = self.TextCNN(last_hidden_state, constant_embedding)
        output = self.af(self.dropout(self.sst_1(output)))
        output = self.sst_2(output)

        y = self.sst_fc1(pooler_output)
        y = self.dropout(y)
        y = self.sst_fc2(y)

        output = (y + output) / 2

        return output

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO

        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)
        pooler_1 = embedding_1['pooler_output']
        pooler_2 = embedding_2['pooler_output']
        hidden_1 = embedding_1['last_hidden_state'] * attention_mask_1.unsqueeze(-1)
        hidden_2 = embedding_2['last_hidden_state'] * attention_mask_2.unsqueeze(-1)
        constant_embedding_1 = self.bert.embed(input_ids_1) * attention_mask_1.unsqueeze(-1)
        constant_embedding_2 = self.bert.embed(input_ids_2) * attention_mask_2.unsqueeze(-1)

        core_1 = self.TextCNN_para(hidden_1, constant_embedding_1)
        core_2 = self.TextCNN_para(hidden_2, constant_embedding_2)
        core = torch.cat((core_1, core_2), dim=1)
        output = self.af(self.dropout(self.para_1(core)))
        output = self.af(self.dropout(self.para_2(output)))
        output = self.af(self.dropout(self.para_3(output)))

        x3 = torch.abs(pooler_1 - pooler_2)
        x = torch.cat((pooler_1, pooler_2, x3), dim=1)

        x = self.para_fc1(x)
        x = self.dropout(x)
        x = self.para_fc2(x)
        x = self.dropout(x)
        x = self.para_fc3(x)

        output = (x + output) / 2

        return output

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        hidd_1 = self.forward(input_ids_1, attention_mask_1)['last_hidden_state'] * attention_mask_1.unsqueeze(-1)
        hidd_2 = self.forward(input_ids_2, attention_mask_2)['last_hidden_state'] * attention_mask_2.unsqueeze(-1)
        hidd = torch.cat((hidd_1, hidd_2), dim=1)
        hidd = self.dropout(self.af(self.similarity_4(hidd)))
        hidd = hidd.squeeze(2)
        hidd = torch.mean(hidd, dim=1).unsqueeze(1)

        out_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
        out_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']
        out_1 = self.similarity_1(out_1)
        out_2 = self.similarity_2(out_2)
        out = torch.cat((out_1, out_2), 1)
        out = self.similarity_3(out)
        final = torch.cat((out, hidd), dim=1)
        final = self.af(self.similarity_5(final))
        return final


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


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train,
                                                                                      args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev,
                                                                                args.sts_dev, split='train')

    # Load data for sentiment analysis
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Load data for paraphrase detection
    para_train_data = SentencePairDataset(para_train_data[:5000], args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)

    # Load data for semantic textual similarity
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_train_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()

        # Enable all tasks
        task = {
            'sst': True,
            'para': False,
            'sts': False,
        }

        if task['sst']:  # Train sentiment analysis
            train_sst_loss = 0
            num_batches = 0
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}-sst', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                           batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_sst_loss += loss.item()
                num_batches += 1

            train_sst_loss = train_sst_loss / (num_batches)

        if task['para']:  # Train paraphrase detection
            train_para_loss = 0
            num_batches = 0
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}-para', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                                  batch['token_ids_2'], batch['attention_mask_2'],
                                                                  batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = logits.to(device)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.unsqueeze(1).float(), reduction='mean')

                loss.backward()
                optimizer.step()

                train_para_loss += loss.item()
                num_batches += 1

            train_para_loss = train_para_loss / (num_batches)

        if task['sts']:  # Train semantic textual similarity
            train_sts_loss = 0
            num_batches = 0
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}-sts', disable=TQDM_DISABLE):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                                                                  batch['token_ids_2'], batch['attention_mask_2'],
                                                                  batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = logits.to(device)

                loss_function = nn.MSELoss()
                loss = loss_function(logits, b_labels.unsqueeze(1).float())
                # loss = F.binary_cross_entropy_with_logits(logits, b_labels.unsqueeze(1).float() / 5.0, reduction='mean')

                loss.backward()
                optimizer.step()

                train_sts_loss += loss.item()
                num_batches += 1

            train_sts_loss = train_sts_loss / (num_batches)

        if task['sst']:  # Evaluate sentiment analysis
            train_sst_acc, *_ = model_eval_sst(sst_train_dataloader, model, device)
            dev_sst_acc, *_ = model_eval_sst(sst_dev_dataloader, model, device)
            dev_acc = dev_sst_acc
            print(
                f"Epoch {epoch} [SST]: train loss :: {train_sst_loss:.3f}, train acc :: {train_sst_acc:.3f}, dev acc :: {dev_sst_acc:.3f}")

        if task['para']:  # Evaluate paraphrase detection
            train_para_acc, *_ = model_eval_para(para_train_dataloader, model, device)
            dev_para_acc, *_ = model_eval_para(para_dev_dataloader, model, device)
            dev_acc = dev_para_acc
            print(
                f"Epoch {epoch} [PARA]: train loss :: {train_para_loss:.3f}, train acc :: {train_para_acc:.3f}, dev acc :: {dev_para_acc:.3f}")

        if task['sts']:  # Evaluate semantic textual similarity
            train_sts_corr, *_ = model_eval_sts(sts_train_dataloader, model, device)
            dev_sts_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device)
            dev_acc = dev_sts_corr
            print(
                f"Epoch {epoch} [STS]: train loss :: {train_sts_loss:.3f}, train corr :: {train_sts_corr:.3f}, dev corr :: {dev_sts_corr:.3f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


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
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--local_files_only", action='store_true')
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--max_position_embeddings", type=int, default=768)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
