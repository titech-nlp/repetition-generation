import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scripts.utils import JsonlDataset
from torch.utils.data import DataLoader
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)


class WeightedLabelSmoothingCrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.CE = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, preds, target, args, source_ids, scores):
        ux = None
        for ids, score in zip(source_ids, scores):
            ux_one = torch.zeros(preds.size(2))  # 32128
            for id, sco in zip(ids, score):
                if sco < 0:
                    break
                ux_one[id] = sco

            ux_one = torch.stack((ux_one, ux_one))
            for _ in range(5):  # 64
                ux_one = torch.cat((ux_one, ux_one))
            if ux == None:
                ux = ux_one.view(1, ux_one.size()[0], ux_one.size()[1])
            else:
                ux = torch.cat(
                    (ux, ux_one.view(1, ux_one.size()[0], ux_one.size()[1])))

        ux = ux.cuda(args.gpu_id)

        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        ux = ux.view(-1, ux.size(-1))

        ori_loss = self.CE(preds, target)

        log_preds = F.log_softmax(preds, dim=-1)
        ls_loss = -(ux*log_preds).sum(dim=1).mean()

        loss = (1 - self.epsilon) * ori_loss + self.epsilon * ls_loss
        loss = ori_loss + self.epsilon * ls_loss
        return loss


class T5FineTuner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_t5_model)

        self.tokenizer = T5Tokenizer.from_pretrained(
            args.pretrained_t5_model, is_fast=True)

        self.loss_fct = WeightedLabelSmoothingCrossEntropyLoss(
            epsilon=args.epsilon, ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        # calculate loss
        labels = batch['target_ids']
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        # overwrite the loss
        loss = self.loss_fct(outputs.logits, labels, self.args,
                             batch['source_ids'], batch['scores'])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('test_loss', loss)
        return {'test_loss': loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr,
                          eps=1e-8,
                          no_deprecation_warning=True)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.t_total)
        self.scheduler = scheduler

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = JsonlDataset(
                self.tokenizer, args.data_dir, "train", alpha=args.alpha)

            self.val_dataset = JsonlDataset(
                self.tokenizer, args.data_dir, "valid", alpha=args.alpha)

            self.t_total = ((len(self.train_dataset) // (self.args.batch_size)) //
                            self.args.gradient_accumulation_steps * float(self.args.epoch))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.args.batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=4)


def main(args):
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=[args.gpu_id],
        max_epochs=args.epoch,
        gradient_clip_val=1.0,
        default_root_dir=args.model_dir
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    os.makedirs(args.model_dir, exist_ok=True)
    model.tokenizer.save_pretrained(args.model_dir)
    model.model.save_pretrained(args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu-id', default=0, type=int)
    parser.add_argument('-data-dir',
                        default='data/', type=str)
    parser.add_argument('-model-dir', default='models/', type=str)
    parser.add_argument('-pretrained-t5-model',
                        default='sonoisa/t5-base-japanese', type=str)
    parser.add_argument('-epsilon', default=0.1, type=float)
    parser.add_argument('-epoch', default=8, type=int)
    parser.add_argument('-alpha', default=1, type=float)
    parser.add_argument('-weight-decay', default=0.0, type=float)
    parser.add_argument('-lr', default=3e-4, type=float)
    parser.add_argument('-batch-size', default=8, type=int)
    parser.add_argument('-gradient-accumulation-steps', default=1, type=int)
    args = parser.parse_args()

    main(args)
