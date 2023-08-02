# -*- coding: utf-8 -*-
# @Time:  0:13
# @Author: tk
# @File：modelweighter
import os
import re
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from deep_training.trainer.pl.modelweighter import *



__all__ = [
    'ModelWeightMixin',
    'ModelWeightMinMax',
    'LoraModel',
    'LoraArguments',
    'LoraConfig',
    'AutoConfig',
    'PromptLearningConfig',
    'PromptModel',
    'PromptArguments',
    'get_prompt_model',
    'ModelArguments',
    'TrainingArguments',
    'DataArguments',
    'PreTrainedModel',
    'HfArgumentParser'
]


ModelWeightMixin_ = ModelWeightMixin

# class ModelWeightMixin(ModelWeightMixin_):
#     def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False, adapter_name="default"):
#         assert os.path.exists(sft_weight_path)
#         if self.lora_args is not None and self.lora_args.with_lora:
#             # 恢复权重
#             self.backbone: LoraModel
#             self.backbone.load_adapter(sft_weight_path, adapter_name=adapter_name, is_trainable=is_trainable,
#                                        strict=strict)
#
#         elif self.prompt_args is not None and self.prompt_args.with_prompt:
#             # 恢复权重
#             self.backbone: PromptModel
#             self.backbone.load_adapter(sft_weight_path, adapter_name=adapter_name, is_trainable=is_trainable,
#                                        strict=strict)
#         else:
#             weight_dict = torch.load(sft_weight_path)
#             weights_dict_new = OrderedDict()
#             valid_keys = ['module', 'state_dict']
#             for k in valid_keys:
#                 if k in weight_dict:
#                     weight_dict = weight_dict[k]
#                     break
#             pl_model_prefix = '_TransformerLightningModule__backbone'
#             is_pl_weight = pl_model_prefix in ','.join(list(weight_dict.keys()))
#             base_model_prefix = self.backbone.base_model_prefix
#             model_prefix = r'{}\.{}'.format(pl_model_prefix, base_model_prefix)
#             for k, v in weight_dict.items():
#                 if is_pl_weight:
#                     k = re.sub(r'_forward_module\.', '', k)
#                     # llm module
#                     if k.startswith(model_prefix):
#                         k = re.sub(r'{}\.'.format(model_prefix), '', k)
#                         k = model_prefix + '.' + k
#                     # TransformerBase module
#                     if not k.startswith(pl_model_prefix):
#                         k = pl_model_prefix + '.' + k
#                 else:
#                     # hf module weight
#                     k = model_prefix + '.' + k
#                 weights_dict_new[k] = v
#
#             # 加载sft 或者 p-tuning-v2权重
#             def assert_state_dict_fn(model, incompatible_keys: _IncompatibleKeys):
#                 if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
#                     return None
#                 _keys_to_ignore_on_load_missing = getattr(model.backbone.model, "_keys_to_ignore_on_load_missing", [])
#                 missing_keys = [_ for _ in incompatible_keys.missing_keys]
#                 model_prefix = r'{}\.{}\.'.format(pl_model_prefix, base_model_prefix)
#                 missing_keys = [re.sub(r'{}'.format(model_prefix), '', _) for _ in missing_keys]
#                 missing_keys = [re.sub(r'{}'.format(pl_model_prefix), '', _) for _ in missing_keys]
#                 if missing_keys and _keys_to_ignore_on_load_missing:
#                     __ = []
#                     for _ in _keys_to_ignore_on_load_missing:
#                         for missing_key in missing_keys:
#                             if re.match(re.compile(_), missing_key):
#                                 __.append(missing_key)
#                     for _ in __:
#                         missing_keys.remove(_)
#
#                 if missing_keys:
#                     if strict:
#                         raise ValueError('Error in load_sft_weight missing_keys', missing_keys)
#                     else:
#                         print('Error in load_sft_weight missing_keys', missing_keys)
#                 if incompatible_keys.unexpected_keys:
#                     if strict:
#                         raise ValueError('Error in load_sft_weight unexpected_keys', incompatible_keys.unexpected_keys)
#                     else:
#                         print(('Error in load_sft_weight unexpected_keys', incompatible_keys.unexpected_keys))
#
#                 if not missing_keys and not incompatible_keys.unexpected_keys:
#                     return None
#                 return missing_keys or incompatible_keys.unexpected_keys
#
#             self: nn.Module
#             h = self.register_load_state_dict_post_hook(assert_state_dict_fn)
#             # TransformerBase类 可能有自定义额外模块
#             self.load_state_dict(weights_dict_new, strict=strict)
#             h.remove()



ModelWeightMinMax = ModelWeightMixin

