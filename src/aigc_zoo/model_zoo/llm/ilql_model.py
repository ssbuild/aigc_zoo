# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/29 9:52
import torch
from deep_training.nlp.rl.ilql.configuration import ILQLArguments, ILQLConfig
from deep_training.nlp.rl.ilql.ilql_module import ILQLModelLoss
from deep_training.nlp.models.rl.modeling_ilql import AutoModelForCausalLMWithILQLHeads
from transformers import AdamW

from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)

class ILQLModelForCausalLMWithILQLHeads(AutoModelForCausalLMWithILQLHeads):
    def __init__(self, *args,hidden_size=None, up_sampling_score=False,**kwargs):
        config = kwargs.get('config')
        if hidden_size is None:
            hidden_size = config.word_embed_proj_dim if getattr(config, 'word_embed_proj_dim', None) else config.hidden_size
        super(ILQLModelForCausalLMWithILQLHeads, self).__init__(*args,hidden_size=hidden_size,up_sampling_score=up_sampling_score, **kwargs)

    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()


class MyILQLTransformer(ILQLModelForCausalLMWithILQLHeads, ILQLModelLoss,ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args, new_num_tokens = None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        ilql_args: ILQLConfig = kwargs.pop('ilql_args', None)
        if ilql_args is not None:
            kwargs.update({
                "two_qs": ilql_args.two_qs,
                "alpha": ilql_args.alpha,
            })
        super(MyILQLTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ilql_config = ilql_args
        self.prompt_args = prompt_args

        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.inject_model()


    def inject_model(self):
        lora_args = self.lora_args
        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: PetlModel = PetlModel(self.backbone, lora_args,
                                         auto_prepare_kbit_training=getattr(self,"auto_prepare_kbit_training",True), 
                                         use_gradient_checkpointing=getattr(self, "gradient_checkpointing", False)
                                         )
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
        return super(MyILQLTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self,*args,**kwargs):
        return self.get_llm_model().generate(*args,**kwargs)

    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        training_args = self.training_args
        optimizer = AdamW(p, lr=training_args.learning_rate,
                          eps=training_args.adam_epsilon,
                          betas=training_args.optimizer_betas,
                          weight_decay=training_args.weight_decay)
        return optimizer


    def training_step(self,*args, **inputs):
        outputs = self.compute_loss(*args, **inputs)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        return self.forward_ilql_loss(*args, **inputs)


    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)