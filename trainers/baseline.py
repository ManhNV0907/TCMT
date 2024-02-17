# learning without forgetting
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import get_linear_schedule_with_warmup
from model.baseline import Baseline
from utils.read_data import statistic


class Trainer:

    def __init__(self, args):
        self.args = args
        if args.method == "baseline":
            self.model = Baseline(args)
        self.task_num = 0

    def new_task(self, train_dataset, test_dataset, num_labels):
        self.task_num += 1
        # expand classifier and prefix
        self.model.new_task(num_labels)
        # fit to new dataset
        self.training(train_dataset)
        # self.training(train_dataset, test_dataset)
        # evaluating
        self.evaluating(test_dataset)

    def training(self, dataset, test_dataset=None):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr_list[self.task_num - 1], weight_decay=0.0)
        num_training_steps = len(
            loader) * self.args.epochs_list[self.task_num - 1]
        num_warmup_steps = num_training_steps * self.args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)

        self.model.cuda()
        for epoch in range(self.args.epochs_list[self.task_num - 1]):
            self.model.train()
            correct, total = 0, 0
            total_loss = 0
            for idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
                outputs = self.optimization(batch, optimizer, scheduler)
                total_loss += outputs.loss.item()

                _, _, labels = batch
                labels = labels.cuda()
                pred = torch.argmax(outputs.logits, dim=1)
                correct += torch.sum(pred == labels).item()
                total += len(labels)


            print(f"Epoch {epoch} Training Accuracy: {correct/total}")
            print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
            if test_dataset is not None:
                self.evaluating(test_dataset)

    def optimization(self, batch, optimizer, scheduler):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        return outputs

    def evaluating(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _, batch in enumerate(tqdm(loader, desc="Evaluating")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1)
                correct += torch.sum(pred == labels).item()
                total += len(labels)

            print(f"Evaluating Accuracy: {correct/total: .4f}")
        return correct / total

    def evaluating_for_datsets(self, datasets):
        eval_acc = [self.evaluating(dataset)
                    for _, dataset in enumerate(datasets)]
        print(f"-" * 50)
        print(f"Average Evaluating Accuracy: {np.mean(eval_acc): .4f}")
        print(f"{np.around(eval_acc, 4)}")
        print(f"-" * 50)
        return eval_acc

   