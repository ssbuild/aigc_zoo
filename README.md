## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    11-13 0.2.9
    11-01 0.2.7 
          0.2.7.post1 support skywork,bluelm
          0.2.7.post2 support yi
    10-19 0.2.6 support visualglm qwen_vl muti-model
    10-12 0.2.5.post5 gradient_checkpointing = False
    09-26 0.2.4 support qwen-7b 新版 和 qwen-14b ， 旧版不再支持，旧版可以安装 deep_training <= 0.2.3
                support transformers trainer
    09-22 0.2.3 support auto finetuning and dpo finetuning
    09-06 0.2.2 调整baichuan模块命名，支持百川v2
    09-03 0.2.1 fix llama2 
    08-25 0.2.0.post1 support xverse chat
    08-23 0.2.0 release
    08-16 0.1.21 release add 5 kind rope scale method
    08-11 0.1.17.post0 update qwen config
    08-09 0.1.17 release
    08-08 0.1.15.rc0 support xverse-13b
    08-05 0.1.13.post0 fix some bugs
    07-18 support InternLM 
    07-16 support rwkv world and fix some bugs
    07-11 support generate for score , support baichuan2
    07-07 fix https://github.com/ssbuild/aigc_zoo/issues/1
    06-16 initial aigc_zoo
```

## install
  - 源码安装
```text
 pip uninstall aigc_zoo
 pip install -U git+https://github.com/ssbuild/aigc_zoo.git
 
 或者 
 git clone https://github.com/ssbuild/aigc_zoo.git
 pip install -e .
```


```text
推荐环境 
python >= 3.10
torch >= 2
```

## AIGC 数据共享
[数据共享](http://124.70.99.221:8080)
[大模型评估](https://github.com/ssbuild/aigc_evals)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [moss_finetuning](https://github.com/ssbuild/moss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [skywork_finetuning](https://github.com/ssbuild/skywork_finetuning)
- [bluelm_finetuning](https://github.com/ssbuild/bluelm_finetuning)
- [yi_finetuning](https://github.com/ssbuild/yi_finetuning)
- [aigc_evals](https://github.com/ssbuild/aigc_evals)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)



## 
    纯粹而干净的代码



## 训练经验分享
[Chatglm2-lora微调](https://blog.csdn.net/feifeiyechuan/article/details/131458322) <br>
[DoctorGLM: 基于ChatGLM-6B的中文问诊模型](https://modelnet.ai/modeldoc/bb2aac4ba2a44f0b96af958f10f57ec4)