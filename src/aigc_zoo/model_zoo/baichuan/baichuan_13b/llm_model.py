# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
import typing
from typing import Optional, List,Union,Any
import torch
from deep_training.nlp.models.baichuan_13b.modeling_baichuan import BaichuanForCausalLM,TransformerBaichuanLMHeadModel,BaichuanConfig,setup_model_profile
from deep_training.nlp.models.transformer_base import TransformerBase
from transformers import GenerationConfig

from ....utils.transformer_utils import hf_decorator
from ....weight.modelweighter import *
from .tokenization_baichuan import BaichuanTokenizer
import logging
logger = logging.getLogger(__name__)


class MyBaichuanForCausalLM(BaichuanForCausalLM):
    def _build_chat_input(self, tokenizer, messages: List[dict], max_new_tokens: int=0):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = self.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.config.model_max_length // 2, max_input_tokens)
        total_input, round_input = [], []
        user_token_id = getattr(self.generation_config,'user_token_id',None)
        assistant_token_id = getattr(self.generation_config,'assistant_token_id',None)
        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [user_token_id] + content_tokens + round_input if user_token_id is not None else content_tokens + round_input
                if total_input and len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message['role'] == 'assistant':
                round_input = [assistant_token_id] + content_tokens + round_input if assistant_token_id is not None else content_tokens + round_input
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        if assistant_token_id is not None:
            total_input.append(assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.device)
        return total_input

    @torch.no_grad()
    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None,**kwargs):
        generation_config = generation_config or self.generation_config
        input_ids = self._build_chat_input(tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
            self.__class__.generate = NewGenerationMixin.generate
            self.__class__.sample_stream = NewGenerationMixin.sample_stream
            stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True,**kwargs)

            def stream_generator():
                outputs = []
                for token in self.generate(input_ids, generation_config=stream_config,**kwargs):
                    outputs.append(token.item())
                    yield tokenizer.decode(outputs, skip_special_tokens=True)

            return stream_generator()
        else:
            self.__class__.generate = PreTrainedModel.generate  # disable stream
            outputs = self.generate(input_ids, generation_config=generation_config,**kwargs)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response


class TransformerBaichuanModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerBaichuanModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyBaichuanForCausalLM, *args, **kwargs))


class TransformerForLM(TransformerBaichuanModel):
    def __init__(self, *args, **kwargs):
        super(TransformerForLM, self).__init__(*args, **kwargs)

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
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        #可能扩充词表
        self.resize_token_embs(new_num_tokens)
        self.inject_model()


    def inject_model(self):
        lora_args, prompt_args = self.lora_args, self.prompt_args
        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: PetlModel = PetlModel(self.backbone, lora_args,auto_prepare_kbit_training=True,use_gradient_checkpointing=getattr(self,"gradient_checkpointing",False) )
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


    def get_llm_model(self) -> Optional[Union[BaichuanForCausalLM,Any]]:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model





