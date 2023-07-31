# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/16 16:37

from typing import List, Tuple
import torch
from transformers import PreTrainedModel,PreTrainedTokenizer

class Generate:
    @classmethod
    @torch.no_grad()
    def generate(cls,model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        output_scores = gen_kwargs.get('output_scores', False)
        if output_scores:
            gen_kwargs['return_dict_in_generate'] = True
        # prompt = "Human：" + query + "\nAssistant："

        prompt = query
        if isinstance(tokenizer,PreTrainedTokenizer):
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            input_len = len(inputs["input_ids"][0])
        else:
            inputs = tokenizer.encode(prompt)
            inputs = torch.tensor([inputs],dtype=torch.int64)
            inputs = inputs.to(model.device)
            outputs = model.generate(inputs, **gen_kwargs)
            input_len = len(inputs[0])
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][input_len:]
        response = tokenizer.decode(outputs)
        return response

    @classmethod
    @torch.no_grad()
    def chat(cls, model: PreTrainedModel, tokenizer, query: str, history: List[Tuple[str, str]] = None, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []

        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        output_scores = gen_kwargs.get('output_scores', False)
        if output_scores:
            gen_kwargs['return_dict_in_generate'] = True

        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        if isinstance(tokenizer, PreTrainedTokenizer):
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            input_len = len(inputs["input_ids"][0])
        else:
            inputs = tokenizer.encode(prompt)
            inputs = torch.tensor([inputs], dtype=torch.int64)
            inputs = inputs.to(model.device)
            outputs = model.generate(inputs, **gen_kwargs)
            input_len = len(inputs[0])
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][input_len:]
        response = tokenizer.decode(outputs)
        history = history + [(query, response)]
        return response, history