# -*- coding: utf-8 -*-
# @Time:  19:00
# @Author: tk
# @File：llm_model

import copy
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable, Generator, Any, Union
import torch
from deep_training.nlp.models.qwen.modeling_qwen import QWenConfig,QWenLMHeadModel,setup_model_profile
from deep_training.nlp.models.transformer import TransformerBase
from torch import nn
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig, StoppingCriteriaList, \
    PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput

from .qwen_generation_utils import HistoryType, make_context, get_stop_words_ids, decode_tokens, \
    StopWordsLogitsProcessor
from .tokenization_qwen import QWenTokenizer
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)

class MyQWenLMHeadModel(QWenLMHeadModel):
    def __init__(self,config):
        super(MyQWenLMHeadModel, self).__init__(config)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(self.generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=self.generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    def chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        **kwargs
    ) -> Tuple[str, HistoryType]:

        if history is None:
            history = []

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=6144,
            chat_format=self.generation_config.chat_format,
        )

        if "stop_words_ids" not in kwargs:
            stop_words_ids = get_stop_words_ids(
                self.generation_config.chat_format, tokenizer
            )
            kwargs['stop_words_ids'] = stop_words_ids
        input_ids = torch.tensor([context_tokens]).to(self.device)

        outputs = self.generate(
            input_ids,
            return_dict_in_generate=False,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=self.generation_config.chat_format,
            verbose=False,
        )

        if append_history:
            history.append((query, response))

        return response, history

    def chat_stream(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            **kwargs
    ) -> Generator[str, Any, None]:

        if history is None:
            history = []

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=6144,
            chat_format=self.generation_config.chat_format,
        )

        if 'stop_words_ids' not in kwargs:
            stop_words_ids = get_stop_words_ids(
                self.generation_config.chat_format, tokenizer
            )
            kwargs['stop_words_ids'] = stop_words_ids
        input_ids = torch.tensor([context_tokens]).to(self.device)

        assert self.generation_config.chat_format == 'chatml'
        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**self.generation_config.to_dict(), **kwargs, do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.generate(input_ids, return_dict_in_generate=False, generation_config=stream_config):
                outputs.append(token.item())
                if outputs[-1] in (tokenizer.im_end_id, tokenizer.im_start_id):
                    break
                yield tokenizer.decode(outputs, skip_special_tokens=True)

        return stream_generator()

class MyTransformerForQwen(TransformerBase):
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
        super(MyTransformerForQwen, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyQWenLMHeadModel, *args, **kwargs))

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
        pass
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.enable_input_require_grads()









class MyTransformer(MyTransformerForQwen,ModelWeightMixin, with_pl=True):
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args',None)
        num_layers_freeze = kwargs.pop('num_layers_freeze',-1)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args

        #可能添加新词
        self.resize_token_embs(new_num_tokens)

        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model = LoraModel(self.backbone, lora_args)
            print('==' * 30,'lora info')
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

        elif num_layers_freeze > 0 and self.config.pre_seq_len is None:  # 非 lora freeze 非 ptuning模式
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'),param[0])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < num_layers_freeze:
                        param[1].requires_grad = False
                        print('freeze layer',param[0])

    def resize_token_embs(self, new_num_tokens):
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
                model.resize_token_embeddings(new_num_tokens)
                # print('after',self.config)

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.with_lora:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyQWenLMHeadModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model