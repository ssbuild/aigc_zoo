# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 14:47

import re
from deep_training.nlp.layers.rope_scale.patch import *
from typing import List, Tuple
import torch
from torch import nn
from deep_training.nlp.models.moss import MossForCausalLM,MossConfig # noqa
from deep_training.nlp.models.moss.tokenization_moss import MossTokenizer # noqa
from deep_training.nlp.models.transformer import TransformerBase

from ...utils.dpo_utils import DpoModule
from ...weight.modelweighter import *
from ...utils.transformer_utils import hf_decorator
import logging
logger = logging.getLogger(__name__)

class MyMossForCausalLM(MossForCausalLM):
    def __init__(self,config):
        super(MyMossForCausalLM, self).__init__(config)
        # self.transformer.gradient_checkpointing = True


class TransformerDPOForLM(DpoModule,TransformerBase):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False, **kwargs):
        super(TransformerDPOForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyMossForCausalLM, *args, **kwargs))
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
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()




class TransformerDPO(TransformerDPOForLM,ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args',None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        num_layers_freeze = kwargs.pop('num_layers_freeze', -1)
        super(TransformerDPO, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.num_layers_freeze = num_layers_freeze

        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        # 可能扩充词表
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.inject_model()

    def inject_model(self):
        lora_args,prompt_args = self.lora_args,self.prompt_args
        num_layers_freeze = self.num_layers_freeze
        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: PetlModel = PetlModel(self.backbone, lora_args, auto_prepare_kbit_training=True)
            print('==' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
            # for name, module in model.named_modules():
            #     if isinstance(module, LoraLayer):
            #         module = module.to(torch.bfloat16)
            #     if 'norm' in name:
            #         module = module.to(torch.float32)
            #     if 'lm_head' in name or 'embed_tokens' in name:
            #         if hasattr(module, 'weight'):
            #             if module.weight.dtype == torch.float32:
            #                 module = module.to(torch.bfloat16)
        elif prompt_args is not None and prompt_args.with_prompt:
            self.backbone.enable_input_require_grads()
            model: PromptModel = get_prompt_model(self.backbone, prompt_args)
            print('==' * 30, 'prompt info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
        elif num_layers_freeze > 0:  # 非 lora freeze
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'), param[0])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < num_layers_freeze:
                        param[1].requires_grad = False
                        print('freeze layer', param[0])

    def resize_token_embs(self, new_num_tokens,pad_to_multiple_of=128):
        if new_num_tokens is not None:
            logger.info(f"new_num_tokens:{new_num_tokens}")
            model: PreTrainedModel = self.backbone.model
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if new_num_tokens > embedding_size:
                # lora ptv2 二次加载权重需备份原此词表
                if (self.lora_args is not None and self.lora_args.with_lora) or (
                        self.prompt_args is not None and self.prompt_args.with_prompt):
                    config = model.config
                    if config.task_specific_params is None:
                        config.task_specific_params = {}
                    config.task_specific_params['vocab_size'] = config.vocab_size

                logger.info("resize the embedding size by the size of the tokenizer")
                # print('before',self.config)
                model.resize_token_embeddings(new_num_tokens,pad_to_multiple_of=pad_to_multiple_of)
                # print('after',self.config)

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.with_lora:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.with_prompt:
            return [(self.backbone, lr)]
        return super(TransformerDPO, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyMossForCausalLM:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model
