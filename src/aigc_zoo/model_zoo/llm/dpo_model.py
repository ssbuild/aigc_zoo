# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
from typing import Dict, Tuple, Union, List
import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.transformer import TransformerForCausalLM
from deep_training.nlp.losses.loss_dpo import dpo_loss
from torch import nn
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.Tensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

class TransformerDPOForLM(TransformerForCausalLM):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False, **kwargs):
        super(TransformerDPOForLM, self).__init__(*args, **kwargs)
        self.beta=beta
        self.ref_free=ref_free
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
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def set_ref_model(self, ref_model):
        self.ref_model = ref_model

    # def forward_logits(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[
    #     torch.FloatTensor, torch.FloatTensor]:
    #     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
    #
    #        We do this to avoid doing two forward passes, because it's faster for FSDP.
    #     """
    #     concatenated_batch = concatenated_inputs(batch)
    #     all_logits = model(concatenated_batch['concatenated_input_ids'],
    #                        attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
    #     all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
    #     chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
    #     rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
    #     return chosen_logps, rejected_logps

    def forward_logits(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.FloatTensor:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        all_logits = model(batch['input_ids2'],attention_mask=batch['attention_mask2']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, batch['labels'], average_log_prob=False)
        return all_logps


    def compute_loss(self, *args, **batch) -> tuple:
        if self.training:
            inputs = {}
            inputs["input_ids"] = torch.cat((batch['input_ids'], batch['input_ids2']), dim=0)
            inputs["attention_mask"] = torch.cat((batch['attention_mask'], batch['attention_mask2']), dim=0)
            inputs["labels"] = torch.cat((batch['labels'],batch['labels2']),dim=0)
        else:
            inputs = batch
        all_logps = self.forward_logits(model=self.backbone,batch=inputs)

        chosen_logps = all_logps[:batch['input_ids'].shape[0]]
        rejected_logps = all_logps[batch['input_ids'].shape[0]:]

        returns = tuple()
        if self.training:
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = self.forward_logits(model=self.ref_model,batch=batch)
            loss = dpo_loss(chosen_logps, rejected_logps,ref_chosen_logps, ref_rejected_logps,beta=self.beta,reference_free=self.ref_free)
            returns += ({"loss":loss},)
        returns += (chosen_logps, rejected_logps)
        return returns


class MyTransformerDPO(TransformerDPOForLM, ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformerDPO, self).__init__(*args, **kwargs)
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
        return super(MyTransformerDPO, self).get_model_lr(model, lr)


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model

