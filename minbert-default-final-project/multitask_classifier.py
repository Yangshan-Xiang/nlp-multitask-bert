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


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 3
        out_channels = 3
        kernel_size = 3

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.af = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        out = self.conv2(x)
        out = out + residual
        out = self.af(out)
        return out


class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        out_channels = 3

        self.Start = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same')
        self.Res1 = ResidualBlock()
        self.Res2 = ResidualBlock()
        self.Res3 = ResidualBlock()
        self.Res4 = ResidualBlock()
        self.spp1 = nn.AdaptiveMaxPool2d((1, 1))
        self.spp2 = nn.AdaptiveMaxPool2d((2, 2))
        self.spp3 = nn.AdaptiveMaxPool2d((4, 4))

        self.fc = nn.Linear(21 * out_channels, classes)

    def forward(self, hidden_states):
        x = hidden_states.unsqueeze(1)
        x = self.Start(x)
        x = self.Res1(x)
        x = self.Res2(x)
        x = self.Res3(x)
        x = self.Res4(x)
        spp1 = self.spp1(x)
        spp1 = spp1.flatten(1)
        spp2 = self.spp2(x)
        spp2 = spp2.flatten(1)
        spp3 = self.spp3(x)
        spp3 = spp3.flatten(1)
        spp = torch.cat((spp1, spp2, spp3), 1)
        out = self.fc(spp)

        return out


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
        self.CNN = CNN(5)
        self.sentiment = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.combine = nn.Linear(10, 5)
        self.paraphrase_1 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.paraphrase_2 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.paraphrase_3 = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
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
        embeddings = self.forward(input_ids, attention_mask)
        pooler_output = embeddings['pooler_output']
        last_hidden_state = embeddings['last_hidden_state']
        out1 = self.sentiment(self.dropout(pooler_output))
        out2 = self.CNN(last_hidden_state)
        out = torch.cat((out1, out2), 1)
        out = self.combine(out)
        scores = F.softmax(out, dim=1)
        return scores

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO

        out_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
        out_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']
        out_1 = self.dropout(self.paraphrase_1(out_1))
        out_2 = self.dropout(self.paraphrase_2(out_2))
        out = torch.cat((out_1, out_2), 1)
        out = self.paraphrase_3(out)
        out = torch.sigmoid(out)
        return out

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
    para_train_data = SentencePairDataset(para_train_data, args)
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
                loss = F.binary_cross_entropy(logits, b_labels.unsqueeze(1).float(), reduction='mean')

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
            print(
                f"Epoch {epoch} [SST]: train loss :: {train_sst_loss:.3f}, train acc :: {train_sst_acc:.3f}, dev acc :: {dev_sst_acc:.3f}")

        if task['para']:  # Evaluate paraphrase detection
            train_para_acc, *_ = model_eval_para(para_train_dataloader, model, device)
            dev_para_acc, *_ = model_eval_para(para_dev_dataloader, model, device)
            print(
                f"Epoch {epoch} [PARA]: train loss :: {train_para_loss:.3f}, train acc :: {train_para_acc:.3f}, dev acc :: {dev_para_acc:.3f}")

        if task['sts']:  # Evaluate semantic textual similarity
            train_sts_corr, *_ = model_eval_sts(sts_train_dataloader, model, device)
            dev_sts_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device)
            print(
                f"Epoch {epoch} [STS]: train loss :: {train_sts_loss:.3f}, train corr :: {train_sts_corr:.3f}, dev corr :: {dev_sts_corr:.3f}")

        if dev_sst_acc > best_dev_acc:
            best_dev_acc = dev_sst_acc
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
