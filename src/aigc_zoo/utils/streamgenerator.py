# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/26 10:29
from typing import Callable
from transformers import TextStreamer


class GenTextStreamer(TextStreamer):

    def __init__(self,process_token_fn: Callable,fn_args, tokenizer, skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer,skip_prompt,**decode_kwargs)
        self.process_token_fn = process_token_fn
        self.fn_args = fn_args

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        # print(text, flush=True, end="" if not stream_end else None)
        self.process_token_fn(text,stream_end,self.fn_args)