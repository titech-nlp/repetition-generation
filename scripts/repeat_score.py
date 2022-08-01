import unicodedata

import torch
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel, T5Tokenizer


def get_subword_span(args, split_data):
    '''
    annotate words to subwords

    e.g.
    word separation: I like basketball
    subword (tokenize) separation: [CLS] I li ##ke bas ##ket ##ball [SEP]

    word_span = [
        [1,1], # I
        [2,3], # li ##ke 
        [4,6]  # bas ##ket ##ball
    ]
    '''

    tokenizer = BertJapaneseTokenizer.from_pretrained(
        args.pretrained_bert_model, do_lower_case=True)

    for i, d in enumerate(split_data):
        utt: str = d['utt']
        utt_words: list = d['utt_words']

        input: dict = tokenizer.batch_encode_plus(
            [utt], max_length=args.input_max_length, truncation=True, padding='max_length')
        utt_tokens = tokenizer.convert_ids_to_tokens(
            input['input_ids'][0], skip_special_tokens=False)

        word_span = []
        start_idx = 1
        end_idx = 0
        for word in utt_words:
            while 1:
                rm_word = utt_tokens[end_idx + 1].replace('##', '')
                if rm_word in word:
                    word = word.replace(rm_word, '', 1)
                    end_idx += 1
                else:
                    break
            # counter-measure for [UNK]
            if start_idx > end_idx:
                end_idx = start_idx

            word_span.append([start_idx, end_idx])
            end_idx += 1
            start_idx = end_idx

        assert len(word_span) == len(utt_words)

        split_data[i]['input_ids_bert'] = input['input_ids'][0]
        split_data[i]['attention_mask_bert'] = input['attention_mask'][0]
        split_data[i]['word_span'] = word_span
    return split_data


class BertRegression(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.pretrained_bert_model)
        self.span_extractor = SelfAttentiveSpanExtractor(args.hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.classifier = torch.nn.Linear(args.hidden_size, 1)

    def forward(self, ids, mask, span, span_mask):
        top_vec = self.bert(ids, attention_mask=mask)

        span_vec = self.span_extractor(
            top_vec.last_hidden_state, span, span_mask)
        span_vec = span_vec.reshape(-1, self.args.hidden_size)
        logits = self.classifier(span_vec).reshape(-1, 140)
        return logits


def cal_rep_score(args, train_loader, test_loader, score_dic: dict):
    model = BertRegression(args)
    model = model.cuda(args.gpu_id)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.num_epoch):
        loss_train = 0
        for data in train_loader:
            input_ids = data['input_ids_bert'].cuda(args.gpu_id)
            att_mask = data['attention_mask_bert'].cuda(args.gpu_id)
            targets = data['target'].cuda(args.gpu_id)
            span = data['word_span'].cuda(args.gpu_id)
            span_mask = (span[:, :, 0] >= 0).long()  # mask of span

            optimizer.zero_grad()
            outputs = model.forward(input_ids, att_mask, span, span_mask)

            loss = criterion(outputs, targets)
            loss = (loss * span_mask.float()).sum()
            non_zero_elements = span_mask.sum()
            loss = loss / non_zero_elements
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        print('Epoch [{}], train loss: {:.4f}'.format(epoch + 1, loss_train))

    model.eval()
    for data in test_loader:
        with torch.no_grad():
            data_ids = data['id']
            input_ids = data['input_ids_bert'].cuda(args.gpu_id)
            att_mask = data['attention_mask_bert'].cuda(args.gpu_id)
            span = data['word_span'].cuda(args.gpu_id)
            sens_len = data['sens_len']
            span_mask = (span[:, :, 0] >= 0).long()  # mask of span

            outputs = model.forward(input_ids, att_mask, span, span_mask)

            for id, out, word_len in zip(data_ids, outputs.tolist(), sens_len):
                score_dic[id] = out[:word_len]
    return score_dic


def calculate_repeat_score(args, data):
    '''
    calculate repeat score by subword

    To caluculate...
    Training data
    - 5 fold cross validation with training data

    Validation and Test data
    - Train a model with all training data
    '''
    repeat_score_dic = dict()
    dataset_train = []
    dataset_val_test = []
    for d in data:
        target = torch.tensor(d['target'])
        span = torch.tensor(d['word_span'], dtype=torch.int32)
        if d['id'].startswith('train'):
            dataset_train.append({
                'id': d['id'],
                'attention_mask_bert': torch.tensor(d['attention_mask_bert']),
                'input_ids_bert': torch.tensor(d['input_ids_bert']),
                'target': torch.cat((target, torch.full((140 - target.size(0),), -1.0))),
                'word_span': torch.cat((span, torch.full((140 - span.size(0), 2), -1.0, dtype=torch.int32))),
                'sens_len': target.size(0),
            })
        else:
            dataset_val_test.append({
                'id': d['id'],
                'attention_mask_bert': torch.tensor(d['attention_mask_bert']),
                'input_ids_bert': torch.tensor(d['input_ids_bert']),
                'target': torch.cat((target, torch.full((140 - target.size(0),), -1.0))),
                'word_span': torch.cat((span, torch.full((140 - span.size(0), 2), -1.0, dtype=torch.int32))),
                'sens_len': target.size(0),
            })

    kf = KFold(n_splits=5)
    for i, (train_indices, test_indices) in enumerate(kf.split(dataset_train)):
        print(f'Training {i} for training data')
        train, test = [], []
        for idx in train_indices:
            train.append(dataset_train[idx])
        for idx in test_indices:
            test.append(dataset_train[idx])

        train_loader = DataLoader(
            dataset=train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            dataset=test, batch_size=args.batch_size, shuffle=True)
        repeat_score_dic = cal_rep_score(
            args, train_loader, test_loader, repeat_score_dic)

    print('Training for validation and test data')
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_val_test,
                             batch_size=args.batch_size, shuffle=True)
    repeat_score_dic = cal_rep_score(
        args, train_loader, test_loader, repeat_score_dic)

    for i, d in enumerate(data):
        data[i]['repeat_score_bywords'] = repeat_score_dic[d['id']]

    return data


def annotate_word_to_subword(args, data):
    '''
    Annotate a repeat score of each word to its subword of T5 tokenizer to train T5 model

    e.g.
    subword separation: I li ##ke bas ##ket ##ball
    word separation: I like basketball
    repeat score by word: [0.1, 0.3, 0.5]

    get: 
    data['repeat_score']:repeat score by subword, [0.1, 0.3, 0,3, 0.5, 0.5, 0.5]
    data['input_ids']: tokenized ids of T5 tokenizer
    '''

    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_t5_model, is_fast=True)

    for i, d in enumerate(data):
        utt: str = d['utt']
        rep: str = d['rep']
        rs_word: list = d['repeat_score_bywords']
        utt_words: list = d['utt_words']

        input: dict = tokenizer.batch_encode_plus(
            [utt], max_length=args.input_max_length, truncation=True, padding='max_length')
        utt_tokens: list = tokenizer.convert_ids_to_tokens(
            input['input_ids'][0], skip_special_tokens=False)
        utt_tokens = [u for u in utt_tokens if u != "<pad>"]

        # calculate repeat score of each string and get average score by subword
        repeat_score = []
        s = 0
        cnt = 0
        idx_word = 0
        if input["input_ids"][0][0] == 5:  # ignore start token
            idx_subword = 1
            repeat_score.append(0)
        else:
            idx_subword = 0
            utt_tokens[0] = utt_tokens[0][1:]

        # Change some spacial tokens.
        # When tokenizing, some string was changed.
        #special_tokens = [["？", "?"]]
        # for before, after in special_tokens:
        #    utt = utt.replace(before, after)
        utt = unicodedata.normalize("NFKC", utt)

        for string in utt:
            if utt_tokens[idx_subword] == "<unk>":
                repeat_score.append(0)
                idx_subword += 1

            if string in utt_tokens[idx_subword]:
                utt_tokens[idx_subword] = utt_tokens[idx_subword].replace(
                    string, "", 1)
                s += rs_word[idx_word]
                cnt += 1
            else:
                continue

            if not utt_tokens[idx_subword]:
                repeat_score.append(s / cnt)
                s = 0
                cnt = 0
                idx_subword += 1

            if string in utt_words[idx_word]:
                utt_words[idx_word] = utt_words[idx_word].replace(
                    string, "", 1)
            else:
                idx_word += 1
        repeat_score.append(-1)  # for '</s>'
        assert len(repeat_score) == len(utt_tokens)

        # padding
        repeat_score.extend([-1 for _ in range(140 - len(repeat_score))])

        target: dict = tokenizer.batch_encode_plus(
            [rep], max_length=args.target_max_length, truncation=True, padding='max_length')

        data[i]['repeat_score'] = repeat_score
        data[i]['utt_ids'] = input['input_ids'][0]
        data[i]['rep_ids'] = target['input_ids'][0]
        data[i]['utt_att_mask'] = input['attention_mask'][0]
        data[i]['rep_att_mask'] = target['attention_mask'][0]

    return data
