# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 16:53
import typing
from typing import Optional, List,Union,Any
import torch
from deep_training.nlp.models.baichuan2_13b.modeling_baichuan import BaichuanForCausalLM,BaichuanConfig,setup_model_profile
from deep_training.nlp.models.transformer_base import TransformerBase
from transformers import GenerationConfig

from ....utils.dpo_utils import DpoModule
from ....utils.transformer_utils import hf_decorator
from ....weight.modelweighter import *
from .tokenization_baichuan import BaichuanTokenizer
from .generation_utils import build_chat_input
import logging
logger = logging.getLogger(__name__)


class MyBaichuanForCausalLM(BaichuanForCausalLM):
    ...

class TransformerForLM(DpoModule,TransformerBase):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False,**kwargs):
        super(TransformerForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyBaichuanForCausalLM, *args, **kwargs))
        self.beta = beta
        self.ref_free = ref_free
        self.ref_model = ref_model
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
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()



class MyTransformer(TransformerForLM, ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        #可能扩充词表
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.inject_model()



    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.enable:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.enable:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)


    def get_llm_model(self) -> Optional[Union[BaichuanForCausalLM,Any]]:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model


