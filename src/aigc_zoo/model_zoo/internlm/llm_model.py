# coding=utf8
# @Time    : 2023/07/18 10:41
# @Author  : tk
# @FileName: llm_model
from typing import List, Tuple, Optional
import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.internlm.modeling_internlm import InternLMForCausalLM,TransformerInternLMHeadModel,InternLMConfig,setup_model_profile
from deep_training.nlp.models.transformer_base import TransformerBase
from transformers.generation.streamers import BaseStreamer

from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
from .tokenization_internlm import InternLMTokenizer
import logging
logger = logging.getLogger(__name__)


class MyInternLMForCausalLM(InternLMForCausalLM):
    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = []):
        prompt = ""
        for record in history:
            prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return tokenizer([prompt], return_tensors="pt")

    @torch.no_grad()
    def chat(self,
             tokenizer,
             query: str,
             history: List[Tuple[str, str]] = [],
             streamer: Optional[BaseStreamer] = None,
             max_new_tokens: int = 1024,
             do_sample: bool = True,
             temperature: float = 0.8,
             top_p: float = 0.8,
             eos_token_id=(2, 103028),
             **kwargs):
        inputs = self.build_inputs(tokenizer, query, history)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.generate(**inputs,
                                streamer=streamer,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                eos_token_id=list(eos_token_id),
                                **kwargs)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<eoa>")[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self,
                    tokenizer,
                    query: str,
                    history: List[Tuple[str, str]] = [],
                    max_new_tokens: int = 1024,
                    do_sample: bool = True,
                    temperature: float = 0.8,
                    top_p: float = 0.8,
                    eos_token_id=(2, 103028),
                    **kwargs):
        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]
                token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
                if token.strip() != "<eoa>":
                    print(token, end="")

            def end(self):
                print("")

        return self.chat(
            tokenizer=tokenizer,
            query=query,
            streamer=ChatStreamer(tokenizer=tokenizer),
            history=history,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs
        )


class TransformerForLM(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyInternLMForCausalLM, *args, **kwargs))

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



class MyTransformer(TransformerForLM, ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args

        #可能扩充词表
        self.resize_token_embs(new_num_tokens)
        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()


    def inject_model(self):
        lora_args,prompt_args = self.lora_args,self.prompt_args
        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: PetlModel = PetlModel(self.backbone, lora_args,auto_prepare_kbit_training=True)
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

    def resize_token_embs(self,new_num_tokens):
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
        elif self.prompt_args and self.prompt_args.with_prompt:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)


    def get_llm_model(self) -> MyInternLMForCausalLM:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model





