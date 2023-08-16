# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/15 16:36

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
from deep_training.nlp.layers.rope_scale.patch import *


# @dataclass
# class RopeArguments:
#     """
#       inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
#     """
#     method: Optional[str] = field(default=None, metadata={"help": "one of dynamic_scaled_rotary,dynamic_part_ntk_rotary,"
#                                                         "ntk_scaled_rotary,linear_scaled_rotary,part_ntk_scaled_rotary"})
#     base: int = field(default=10000, metadata={"help": "base default 10000"})
#
#
# class RopeScaleMode(Enum):
#     dynamic_scaled_rotary = 0
#     dynamic_part_ntk_rotary= 1
#     ntk_scaled_rotary = 2
#     linear_scaled_rotary = 3
#     part_ntk_scaled_rotary = 4