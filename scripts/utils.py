import json
import os

import MeCab
import torch
from sklearn import preprocessing
from sumeval.metrics.rouge import RougeCalculator
from torch.utils.data import Dataset


class JsonlDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, input_max_len=512, target_max_len=64, alpha=None):
        self.alpha = alpha
        self.input_file_path = os.path.join(
            data_dir, "preprocessed_data.jsonl")
        self.output_file_path = os.path.join(data_dir, 'output.jsonl')
        self.data_type = data_type

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.rouge = RougeCalculator(lang='ja')
        self.mecab = MeCab.Tagger('-O wakati -r mecabrc -d data/ipadic')

        self.inputs = []
        self.info = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['target_ids'].squeeze()

        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()

        scores = self.info[index]["score"]

        return {'source_ids': source_ids, 'source_mask': source_mask,
                'target_ids': target_ids, 'target_mask': target_mask,
                'scores': scores}

    def _build(self):
        with open(self.input_file_path, 'r', encoding='utf-8') as f_json:
            for line in f_json:
                df = json.loads(line)
                if df["id"].startswith(self.data_type):
                    inp = {"input_str": df["utt"],
                           'input_ids': torch.tensor(df['utt_ids']),
                           'attention_mask': torch.tensor(df['utt_att_mask'])}
                    tar = {"target_str": df["rep"],
                           'target_ids': torch.tensor(df['rep_ids']),
                           'attention_mask': torch.tensor(df['rep_att_mask'])}

                    # normalization of repeat score
                    # i.e. man-max scaling
                    sco = df['repeat_score']
                    sco_norm = preprocessing.minmax_scale(sco)

                    # score to the power of alpha
                    sco_norm = torch.from_numpy(sco_norm**self.alpha)

                    self.inputs.append(inp)
                    self.targets.append(tar)
                    self.info.append({
                        "score": sco_norm,
                        "repeated_words": df["repeated_words"]})

    def register_output(self, outputs: list):
        self.outputs = []
        for out in outputs:
            self.outputs.append({"output_str": out})
        assert len(self.inputs) == len(self.outputs)

    def save_output(self, is_show=False):
        with open(self.output_file_path, 'w') as out_file:
            for output, target, input in zip(self.outputs, self.targets, self.inputs):
                df = {"utterance": input["input_str"],
                      "generated_rep": output["output_str"],
                      "target_rep": target["target_str"]}
                json.dump(df, out_file, ensure_ascii=False)
                out_file.write('\n')
                if is_show:
                    print(f'Utterance: {input["input_str"]}')
                    print(f'Generated repetition: {output["output_str"]}')
                    print(f'Target repetition: {target["target_str"]}\n')

    def test_rouge(self):
        # calculate rouge score
        # use this function after "register_output"
        rouge_1, rouge_2, rouge_l = [], [], []

        for out, tar in zip(self.outputs, self.targets):
            out_str = out["output_str"]
            tar_str = tar["target_str"]
            r1 = self.rouge.rouge_n(summary=out_str, references=tar_str, n=1)
            r2 = self.rouge.rouge_n(summary=out_str, references=tar_str, n=2)
            rl = self.rouge.rouge_l(summary=out_str, references=tar_str)
            rouge_1.append(r1)
            rouge_2.append(r2)
            rouge_l.append(rl)

        cnt = len(self.targets)
        print('---------------------------')
        print('ROUGE 1: {:.4f}'.format(sum(rouge_1) / cnt))
        print('ROUGE 2: {:.4f}'.format(sum(rouge_2) / cnt))
        print('ROUGE L: {:.4f}'.format(sum(rouge_l) / cnt))
        print('---------------------------')

    def check_true_repeat(self):
        # count the number of repetition which has true repeated contents word
        # use this function after "register_output"

        references = [out["output_str"] for out in self.outputs]
        true_repeated_words = [info["repeated_words"] for info in self.info]
        assert len(references) == len(true_repeated_words)

        cnt = len(references)
        rep = []
        for words, ref in zip(true_repeated_words, references):
            flag = 0
            s = set(words)
            ref_parse = self.mecab.parse(ref).split()
            for r in ref_parse:
                if r in s:
                    flag = 1
                    break
            rep.append(flag)

        print('True Repetition in conversation {} in {} ({:.2f}%)'.format(
            sum(rep), cnt, sum(rep)/cnt))
