# coding=utf8
# @Time    : 2023/5/12 21:04
# @Author  : tk
# @FileName: moss_model

import torch
from deep_training.nlp.models.moss import MossForCausalLM,MossConfig
from deep_training.nlp.models.moss.tokenization_moss import MossTokenizer
from deep_training.nlp.models.transformer import TransformerBase


class MyMossForCausalLM(MossForCausalLM):
    def __init__(self,config):
        super(MyMossForCausalLM, self).__init__(config)
        # self.transformer.gradient_checkpointing = True




class MyTransformerMossForCausalLM(TransformerBase):
    def __init__(self, *args,**kwargs):
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        if not load_in_4bit:
            quantization_config = kwargs.get("quantization_config", None)
            if quantization_config:
                load_in_4bit = quantization_config.load_in_4bit

        if not load_in_8bit and not load_in_4bit:
            kwargs.pop("device_map", None)
            kwargs.pop("quantization_config", None)
        super(MyTransformerMossForCausalLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyMossForCausalLM, *args, **kwargs))

        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        # class CastOutputToFloat(nn.Sequential):
        #     def forward(self, x):
        #         return super().forward(x).to(torch.float32)
        #
        # self.model.lm_head = CastOutputToFloat(self.model.lm_head)

    def enable_input_require_grads(self):
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()