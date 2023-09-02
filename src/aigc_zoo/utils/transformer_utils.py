# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 9:18
import typing
def hf_decorator(fn):
    def hf_fn(self,*args,**kwargs):
        preprocess_hf_kwargs(kwargs)
        fn(self,*args,**kwargs)
    return hf_fn

def preprocess_hf_kwargs(kwargs: typing.Dict):
    # load_in_8bit = kwargs.get('load_in_8bit', False)
    # load_in_4bit = kwargs.get('load_in_4bit', False)
    # quantization_config = kwargs.get("quantization_config", None)
    # if quantization_config:
    #     load_in_4bit = load_in_4bit or quantization_config.load_in_4bit
    #     load_in_8bit = load_in_8bit or quantization_config.load_in_8bit
    # if not load_in_8bit and not load_in_4bit:
    #     kwargs.pop("device_map", None)
    #     kwargs.pop("quantization_config", None)
    ...