import os
import json
import MeCab
import argparse

from scripts.repeat_score import get_subword_span, calculate_repeat_score, annotate_word_to_subword


def load_dataset(data_type, f_path):
    data = []
    print(f'loading {f_path}...')
    with open(f_path, 'r') as f_json:
        for line in f_json:
            df = json.loads(line)
            df['type'] = data_type
            data.append(df)
    return data


def make_rep_utt_pair(data):
    '''
    make a utterance and repetition pair
    one repetition is used for training data
    all repetition are used for validation and test data
    '''

    split_data = []
    data_counter = {"train": 0, "valid": 0, "test": 0}
    for d in data:
        if d['type'] == 'train':
            split_data.append(
                {
                    'id': f'{d["type"]}_{str(data_counter[d["type"]])}',
                    'utt': d['dialog'][-1]['text'],
                    'rep': d['repetitions'][0],
                })
            data_counter[d["type"]] += 1
        else:
            for rep in d['repetitions']:
                split_data.append(
                    {
                        'id': f'{d["type"]}_{str(data_counter[d["type"]])}',
                        'utt': d['dialog'][-1]['text'],
                        'rep': rep,
                    })
                data_counter[d["type"]] += 1
    return split_data


def find_repeat_words(split_data):
    '''
    find repeated content words in a repetition

    PoS of content words = ['verb', 'noun', 'adjective', 'adverb']
    '''

    me = MeCab.Tagger('-O chasen -r mecabrc -d data/ipadic')
    content_pos = ['動詞', '名詞', '形容詞', '副詞']  # content words

    for i, d in enumerate(split_data):
        utt = d['utt']
        rep = d['rep']

        rep_words = set()
        node = me.parseToNode(rep)
        while node:
            if node.surface:
                rep_words.add(node.surface)
            node = node.next

        utt_words = []
        is_repeated_list = []
        repeated_words = []
        node = me.parseToNode(utt)
        while node:
            word = node.surface
            pos = node.feature.split(',')[0]
            node = node.next
            if word:
                utt_words.append(word)
            else:
                continue

            if word in rep_words and pos in content_pos:
                is_repeated_list.append(1)
                repeated_words.append(word)
            else:
                is_repeated_list.append(0)

        assert len(is_repeated_list) == len(utt_words)

        split_data[i]['utt_words'] = utt_words
        split_data[i]['repeated_words'] = repeated_words
        split_data[i]['target'] = is_repeated_list

    return split_data


def main(args) -> None:
    data = []
    data.extend(load_dataset('train', os.path.join(
        args.data_dir, "repetition", "train.jsonl")))
    data.extend(load_dataset('valid', os.path.join(
        args.data_dir, "repetition", "valid.jsonl")))
    data.extend(load_dataset('test', os.path.join(
        args.data_dir, "repetition", "test.jsonl")))

    split_data = make_rep_utt_pair(data)
    split_data = find_repeat_words(split_data)
    split_data = get_subword_span(args, split_data)
    split_data = calculate_repeat_score(args, split_data)
    preprocessed_data = annotate_word_to_subword(args, split_data)

    with open(os.path.join(args.data_dir, "preprocessed_data.jsonl"), 'w') as outfile:
        for d in preprocessed_data:
            dic = {
                'id': d['id'],
                'utt': d['utt'],
                'rep': d['rep'],
                'repeat_score': d['repeat_score'],
                'repeated_words': d['repeated_words'],
                'utt_ids': d['utt_ids'],
                'rep_ids': d['rep_ids'],
                'utt_att_mask': d['utt_att_mask'],
                'rep_att_mask': d['rep_att_mask'],
            }
            json.dump(dic, outfile, ensure_ascii=False)
            outfile.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', default=0, type=int)
    parser.add_argument('-data_dir', default='data/', type=str)
    parser.add_argument('-pretrained-bert-model',
                        default='cl-tohoku/bert-base-japanese', type=str)
    parser.add_argument('-pretrained-t5-model',
                        default='sonoisa/t5-base-japanese', type=str)
    parser.add_argument('-input-max-length', default=256, type=int)
    parser.add_argument('-target-max-length', default=64, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-num-epoch', default=3, type=int)
    parser.add_argument('-batch-size', default=50, type=int)
    parser.add_argument('-hidden-size', default=768, type=int)
    args = parser.parse_args()

    main(args)
