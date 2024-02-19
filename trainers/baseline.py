# learning without forgetting
import math
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import get_linear_schedule_with_warmup
from model.baseline import Encoder, Classifier
from utils.read_data import statistic
from sklearn.mixture import GaussianMixture

# @torch.no_grad()
# def convert_data_tokens_to_queries(args, data, encoder):
#     data_loader = get_data_loader(args, data, shuffle=False)
#     queries = []
#     print("Forward data...")
#     for (input_ids, attention_mask, labels) in tqdm(data_loader):
#         # tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
#         queries.append(encoder(input_ids))
#     queries = torch.cat(queries, dim=0).cpu()

#     new_data = copy.deepcopy(data)
#     for i in range(len(new_data)):
#         new_data[i]["tokens"] = queries[i]
#     return new_data

class Trainer:

    def __init__(self, args):
        self.args = args
        if args.method == "baseline":
            self.model = Encoder(args)
        self.task_num = 0
        # past classifier
        past_classifier = None
        # Classifier
        # self.model = Baseline(args)
        # self.classifier = Classifier(args)
        self.buffer_distribution = {}
        self.key_mixture = {}
        self.buffer_embedding = {}

    def new_task(self, train_dataset, test_dataset, num_labels):
        self.task_num += 1
        self.curr_label_set = set(train_dataset.labels)
        for i in self.curr_label_set:
            self.buffer_distribution[i] = [] 
            self.key_mixture[i] = 
            self.buffer_embedding[i] = []
        # expand classifier and prefix
        self.model.new_task(num_labels)
        # train classifier with new dataset
        self.training(train_dataset)
        #train classifier with GMM data set
        #distillation
        # self.training(train_dataset, test_dataset)
        # evaluating
        # self.evaluating(test_dataset)

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
        # self.encoder.cuda()
        # self.classifier.cuda()

        # X_embedding = {}
        for idx, batch in enumerate(tqdm(loader, desc=f"Forward current data")):
            print("Forward current data...")
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                get_prelogits=True
            )
            # print(outputs.keys())
            labels = labels.cpu()
            # X_embedding[labels] = []
            for i, _ in enumerate(labels):
                # print(labels[i].item())
                # print(self.buffer_distribution.keys())
                self.buffer_distribution[labels[i].item()].append(outputs[i].cpu())
            # queries.append(outputs.prelogits)
        for label in self.curr_label_set:
            self.buffer_distribution[label] =  torch.cat(self.buffer_distribution[label], dim=0).reshape(-1, 1)
        # queries = torch.cat(queries, dim=0).cpu()
        # new_data = copy.deepcopy(dataset)
            self.key_mixture[label] = GaussianMixture(n_components=1, random_state=42).fit(self.buffer_distribution[label].cpu().detach().numpy())
            # if self.args.gmm_num_components == 1:
            self.key_mixture[label].weights_[0] = 1.0
                # print()

        # for i in range(len(new_data)):
        #     new_data[i]["tokens"] = queries[i]
        # Current encoded data
        # cur_training_encoded = new_data

        # for epoch in range(self.args.epochs_list[self.task_num - 1]):
        #     self.model.train()
        #     correct, total = 0, 0
        #     total_loss = 0
        #     for idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        #         #Finetune classifier on current data
        #         outputs = self.optimization(batch, optimizer, scheduler)

        #         total_loss += outputs.loss.item()

        #         _, _, labels = batch
        #         labels = labels.cuda()
        #         pred = torch.argmax(outputs.logits, dim=1)
        #         correct += torch.sum(pred == labels).item()
        #         total += len(labels)

        #Sample prelogits 
        for i, label in enumerate(self.curr_label_set):
            replay_embedding =  self.key_mixture[label].sample(100 * 256)[0].astype("float32")
            self.buffer_embedding[label].append(replay_embedding)
            # print(replay_embedding)
           
        print(self.buffer_embedding)


            # print(f"Epoch {epoch} Training Accuracy: {correct/total}")
            # print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
            # if test_dataset is not None:
            #     self.evaluating(test_dataset)

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
    
    @torch.no_grad()
    def sample_gmm_data(self, args, encoder, encoded_data, name, task_id):
        """
        :param encoded_data: (List) data of relation
        """

        encoder.eval()
        data_loader = get_data_loader(args, encoded_data, shuffle=False)
        td = tqdm(data_loader, desc=name)

        # output dict
        out = {}

        # x_data
        x_encoded = []

        for (_, tokens, _) in td:
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            x_encoded.append(tokens)
            # x_encoded.append(encoder(tokens)) # When encoded_data is not encoded but is in original format (tokens)
        x_encoded = torch.cat(x_encoded, dim=0)
        key_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_encoded.cpu().detach().numpy())
        if args.gmm_num_components == 1:
            key_mixture.weights_[0] = 1.0

        out["replay_key"] = key_mixture
        return out

   