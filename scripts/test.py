import argparse
import collections

import numpy as np
import torch
from scripts.utils import JsonlDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main(args):
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    trained_model.cuda(args.gpu_id)
    trained_model.eval()

    test_dataset = JsonlDataset(tokenizer, args.data_dir, 'test',
                                input_max_len=args.input_max_length, target_max_len=args.target_max_length, alpha=args.alpha)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, num_workers=4)

    if args.beam_size != 1:
        beam_params = dict(
            num_beams=args.beam_size,
            diversity_penalty=1.0,
            num_beam_groups=args.beam_size,
            num_return_sequences=args.beam_size,
        )
    else:
        beam_params = dict()

    inputs, outputs, targets = [], [], []
    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']

        input_ids = input_ids.cuda(args.gpu_id)
        input_mask = input_mask.cuda(args.gpu_id)

        output = trained_model.generate(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        max_length=args.target_max_length,
                                        temperature=1.0,
                                        repetition_penalty=1.5,
                                        output_scores=True,
                                        output_attentions=True,
                                        return_dict_in_generate=True,
                                        **beam_params
                                        )

        batch_size = input_ids.size(0)

        sequences = output.sequences
        input_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
                      for ids in input_ids]
        target_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                       for ids in batch['target_ids']]
        output_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                       for ids in sequences]

        inputs.extend(input_text)
        targets.extend(target_text)

        if args.beam_size != 1:
            # Sequences scores
            sequences_scores = output.sequences_scores.view(
                batch_size, args.beam_size)

            # Length penalty
            sequences_scores = sequences_scores / \
                ((5 + output.decoder_attentions[-1]
                 [-1].size()[-1]) / 6.0) ** 0.2

            # Coverage penalty
            last_attentions = output.decoder_attentions[-1][-1].mean(dim=1).view(
                batch_size*args.beam_size, output.decoder_attentions[-1][-1].size()[-1])
            cov_pen = -last_attentions.log().mean(-1)

            cov_pen = cov_pen.view(batch_size, args.beam_size)

            # Repeated Scores
            output_reshape = sequences.view(
                (batch_size, args.beam_size, sequences.size(1)))
            output_text = np.reshape(
                output_text, (batch_size, args.beam_size)).tolist()

            repeated_score = []
            for i in range(batch_size):
                rep_score = [0]*args.beam_size
                for j, ids in enumerate(output_reshape[i]):
                    id_score = collections.defaultdict(int)
                    for id, sco in zip(input_ids[i], batch['scores'][i]):
                        if float(sco) < 0:
                            break
                        id_score[int(id)] = float(sco)

                    for id in ids:
                        rep_score[j] += id_score[int(id)]
                        id_score[int(id)] = 0
                repeated_score.append(rep_score)
            repeated_score = torch.tensor(
                repeated_score).cuda(args.gpu_id)
            repeated_score = repeated_score.log()

            final_score = sequences_scores+0.2*cov_pen+repeated_score
            choiced_output = []
            max_ind = torch.argmax(final_score, dim=1)
            for i, ind in enumerate(max_ind):
                choiced_output.append(output_text[i][ind])
            outputs.extend(choiced_output)

        else:
            outputs.extend(output_text)

    test_dataset.register_output(outputs)
    test_dataset.save_output(is_show=True)

    # Calculate metrics
    test_dataset.test_rouge()
    test_dataset.check_true_repeat()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu-id', default=0, type=int)
    parser.add_argument('-data-dir', default='data/', type=str)
    parser.add_argument('-model-dir', default='models/', type=str)
    parser.add_argument('-pretrained-t5-model',
                        default='sonoisa/t5-base-japanese', type=str)
    parser.add_argument('-beam-size', default=5, type=int)
    parser.add_argument('-alpha', default=1, type=float)
    parser.add_argument('-batch-size', default=8, type=int)
    parser.add_argument('-input-max-length', default=256, type=int)
    parser.add_argument('-target-max-length', default=64, type=int)
    args = parser.parse_args()

    main(args)
