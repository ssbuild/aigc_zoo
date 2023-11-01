# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/28 10:09


__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING"
]


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    'moss': ['qkv_proj'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
    'xverse': ["q_proj", "k_proj", "v_proj"],
    'baichuan': ['W_pack'],
    'internlm': ['q_proj','k_proj','v_proj'],
    'qwen': ['c_attn'],
    "clip": [ "q_proj", "v_proj" ],
    "chinese_clip": ["query","value","k_proj","v_proj"],
    "whisper": ["q_proj", "v_proj"],
    "wav2vec2": ["q_proj", "v_proj"],
    "skywork": ["q_proj", "v_proj"],
    "BlueLM": ["q_proj", "v_proj"],

}

TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "gpt_bigcode": ["c_attn"],
    "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],

    "chatglm": ["query_key_value"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    'moss': ['qkv_proj'],
    'cpmant': ['project_q', 'project_v'],
    'rwkv': ['key', 'value', 'receptance'],
    'xverse': ["q_proj","v_proj"],
    'baichuan': ['W_pack'],
    'internlm': ['q_proj',  'v_proj'],
    'qwen': ['c_attn'],
    "clip": [ "q_proj", "v_proj" ],
    "chinese_clip": [ "query", "value", "k_proj", "v_proj" ],
    "whisper": ["q_proj", "v_proj"],
    "wav2vec2": ["q_proj", "v_proj"],
    "skywork": ["q_proj", "v_proj"],
    "BlueLM": ["q_proj", "v_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "t5": ["k", "v", "wo"],
    "mt5": ["k", "v", "wi_1"],
    "gpt2": ["c_attn", "mlp.c_proj"],
    "bloom": ["query_key_value", "mlp.dense_4h_to_h"],
    "roberta": ["key", "value", "output.dense"],
    "opt": ["q_proj", "k_proj", "fc2"],
    "gptj": ["q_proj", "v_proj", "fc_out"],
    "gpt_neox": ["query_key_value", "dense_4h_to_h"],
    "gpt_neo": ["q_proj", "v_proj", "c_proj"],
    "bart": ["q_proj", "v_proj", "fc2"],
    "gpt_bigcode": ["c_attn", "mlp.c_proj"],
    "llama": ["k_proj", "v_proj", "down_proj"],
    "bert": ["key", "value", "output.dense"],
    "deberta-v2": ["key_proj", "value_proj", "output.dense"],
    "deberta": ["in_proj", "output.dense"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],

    "chatglm": ["query_key_value","ffn.dense_4h_to_h"],
    "mpt": ["Wqkv","mlp.down_proj"],
    "codegen": ["qkv_proj","mlp.fc_out"],
    'moss': ['qkv_proj',"mlp.fc_out"],
    'cpmant': ['project_q', 'project_v',"ffn.ffn.w_out"],
    'rwkv': ['key', 'value', 'receptance',"ffn.value"],
    'xverse': ["q_proj", "v_proj","mlp.down_proj"],
    'baichuan': ['W_pack',"mlp.down_proj"],
    'internlm': ['q_proj', 'v_proj',"mlp.down_proj"],
    'qwen': ['c_attn',"mlp.c_proj"],
    "clip": [ "q_proj", "v_proj", "fc2" ],
    "chinese_clip": ["query","value","k_proj","v_proj", "fc2"],
    "skywork": ["k_proj", "v_proj", "down_proj"],
    "BlueLM": ["k_proj", "v_proj", "down_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {
    "t5": ["wo"],
    "mt5": [],
    "gpt2": ["mlp.c_proj"],
    "bloom": ["mlp.dense_4h_to_h"],
    "roberta": ["output.dense"],
    "opt": ["fc2"],
    "gptj": ["fc_out"],
    "gpt_neox": ["dense_4h_to_h"],
    "gpt_neo": ["c_proj"],
    "bart": ["fc2"],
    "gpt_bigcode": ["mlp.c_proj"],
    "llama": ["down_proj"],
    "bert": ["output.dense"],
    "deberta-v2": ["output.dense"],
    "deberta": ["output.dense"],
    "RefinedWeb": ["query_key_value"],
    "RefinedWebModel": ["query_key_value"],
    "falcon": ["query_key_value"],

    "chatglm": ["ffn.dense_4h_to_h"],
    "mpt": [ "mlp.down_proj"],
    "codegen": ["mlp.fc_out"],
    'moss': ["mlp.fc_out"],
    'cpmant': ["ffn.ffn.w_out"],
    'rwkv': ["ffn.value"],
    'xverse': ["mlp.down_proj"],
    'baichuan': ["mlp.down_proj"],
    'internlm': ["mlp.down_proj"],
    'qwen': ["mlp.c_proj"],
    "clip": [ "fc2",  ],
    "chinese_clip": ["fc2",],
    "skywork": ["down_proj"],
    "BlueLM": ["down_proj"],
}
