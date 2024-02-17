import torch
from copy import deepcopy
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from utils.assets import get_prompts_data, mean_pooling


class Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classifier = nn.Linear(
            # self.config.hidden_size, args.num_labels, device=self.device)
            self.config.hidden_size, 0, device=self.device)
        
    

        if self.args.frozen:
            # frozen model's parameters
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_labels = 0
        self.num_tasks = 0
        self.old_model = None


    def new_task(self, num_labels):
        self.old_num_labels = self.num_labels

        self.num_tasks += 1

        with torch.no_grad():
            # expand classifier
            num_old_labels = self.num_labels
            self.num_labels += num_labels
            w = self.classifier.weight.data.clone()
            b = self.classifier.bias.data.clone()
            self.classifier = nn.Linear(
                self.config.hidden_size, self.num_labels, device=self.device)
            self.classifier.weight.data[:num_old_labels] = w
            self.classifier.bias.data[:num_old_labels] = b

    def get_prelogits(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None
    ):
        outputs = self.model(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        last_hidden_state = outputs.last_hidden_state
        prelogits = mean_pooling(last_hidden_state, attention_mask)

        prelogits = self.dropout(prelogits)
        return outputs, prelogits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        get_logits=False,
    ):

        # for code readability
        args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "past_key_values": past_key_values,
        }

        outputs, pooled_output = self.get_prelogits(**args)
        logits = self.classifier(pooled_output)

        if get_logits:
            return logits

        if self.training:
    

            # if self.num_tasks > 1 and self.args.buffer_ratio == 0 and self.args.buffer_size == 0:
            #     logits[:, :self.old_num_labels] = -1e4

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Classifier(nn.Module):
    def __init__(self, args, final_linear=None):
        top_linear = final_linear if final_linear is not None else nn.Linear(args.encoder_output_size, args.rel_per_task * args.num_tasks)
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(args.encoder_output_size * 2, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.encoder_output_size, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            top_linear,
        ).to(args.device)

    def forward(self, x: torch.Tensor):
        return self.head(x)