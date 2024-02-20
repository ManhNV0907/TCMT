import torch
from copy import deepcopy
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from utils.assets import get_prompts_data, mean_pooling


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.classifier = nn.Linear(
        #     # self.config.hidden_size, 38, device=self.device)
        #     self.config.hidden_size, 0, device=self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_labels = 0
        self.num_tasks = 0
        self.old_model = None


    # def new_task(self, num_labels):
    #     self.old_num_labels = self.num_labels

    #     self.num_tasks += 1
    #     # save old model for distillation
    #     if self.num_tasks > 0:
    #         self.old_model = None
    #         self.old_model = deepcopy(self)
    #     with torch.no_grad():
    #         # expand classifier
    #         num_old_labels = self.num_labels
    #         self.num_labels += num_labels
    #         w = self.classifier.weight.data.clone()
    #         b = self.classifier.bias.data.clone()
    #         self.classifier = nn.Linear(
    #             self.config.hidden_size, self.num_labels, device=self.device)
    #         self.classifier.weight.data[:num_old_labels] = w
    #         self.classifier.bias.data[:num_old_labels] = b

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
        raw_embedding = self.model.embeddings(
        input_ids, position_ids, token_type_ids)
        inputs_embeds = raw_embedding
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
        get_prelogits=False,
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
        # logits = self.classifier(pooled_output)

        if get_prelogits:
            return pooled_output

        # if self.training:
    

        #     if self.num_tasks > 1 and self.args.buffer_ratio == 0 and self.args.buffer_size == 0:
        #         logits[:, :self.old_num_labels] = -1e4

        #     if labels is not None:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(
        #             logits.view(-1, logits.shape[-1]), labels.view(-1))
        # else:
        #     loss = None

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Classifier(nn.Module):
    def __init__(self, args, final_linear=None):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.top_linear = nn.Linear(self.config.hidden_size,0)
        self.num_labels = 0
        self.num_tasks = 0
        self.old_model = None
        self.head = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(768, 768, bias=True),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.head(x)
        out = self.top_linear(x)
        return out    

    def new_task(self, num_labels):
        self.old_num_labels = self.num_labels

        self.num_tasks += 1
        # save old model for distillation
        if self.num_tasks > 0:
            self.old_model = None
            self.old_model = deepcopy(self)
        with torch.no_grad():
            # expand classifier
            num_old_labels = self.num_labels
            self.num_labels += num_labels
            w = self.top_linear.weight.data.clone()
            b = self.top_linear.bias.data.clone()
            self.top_linear = nn.Linear(
                self.config.hidden_size, self.num_labels, device=self.device)
            self.top_linear.weight.data[:num_old_labels] = w
            self.top_linear.bias.data[:num_old_labels] = b