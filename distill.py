# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import wandb
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate
import torch.nn.functional as F
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from reparam_module import ReparamModule
import random

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    run_name: str = field(
        default='none',
        metadata={"help": "wandb run name"}
    )
    wandb_project: str = field(
        default='qlora',
        metadata={"help": "wandb project name"}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    save_interval: int = field(default=1, metadata={"help": "the save intervals for expert trajectories"})
    save_dir: str = field(default='./expert_trajectories', metadata={"help": "the save dir for expert trajectories"})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

@dataclass
class DistillationArguments:
    lr_teacher: Optional[float] = field(default=1e-3)
    lr_text: Optional[float] = field(default=1e2)
    lr_label: Optional[float] = field(default=1e2)
    lr_lr: Optional[float] = field(default=1e-4)
    syn_steps: Optional[int] = field(default=5)
    eval_it: Optional[int] = field(default=200)
    Iteration: Optional[int] = field(default=2000)
    max_start_epoch: Optional[int] = field(default=1)
    expert_epochs: Optional[int] = field(default=1)
    epoch_eval_train: Optional[int] = field(default=10)
    load_all: Optional[bool] = field(default=True)
    max_files: Optional[int] = field(default=None)
    max_experts: Optional[int] = field(default=None)
    expert_dir: Optional[str] = field(default="")
    batch_syn: Optional[int] = field(default=4)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        # load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            # load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(filename=dataset_name, format='jsonlines')
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def create_mask(m, n):
    mask = torch.full((m,n), True, dtype=torch.bool)
    if m == 1 or n == 1:
        return mask
    x = random.randint(1, m-1)
    y = random.randint(1, n-1)
    mask[x:, y:] = False
    return mask

def creat_labels(m, n):
    syn_labels = []
    for i in range(0, m):
        labels = torch.randint(3, 32001, (n,))
        a = torch.randint(0, n, (1,)).item()
        b = torch.randint(0, n-a, (1,)).item()

        labels[:a] = -100
        labels[-b:] = -100

        labels[-(b+1)] = 2
        syn_labels.append(labels)
    
    syn_labels = torch.stack(syn_labels, dim=0)
    return syn_labels

        

def distill():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments, DistillationArguments
    ))
    model_args, data_args, training_args, generation_args, distillation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(distillation_args)
    )
    print(args)

    # wandb.init(sync_tensorboard=False,
    #             entity='tongchen',
    #             project="qlora-distill",
    #             name="size=100-lr_teacher={}-lr_text={}-lr_lr={}-syn_steps={}".format(args.lr_teacher, args.lr_text, args.lr_lr, args.syn_steps)
    #            )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    distributed = torch.cuda.device_count() > 1

    # import pdb
    # pdb.set_trace()
    memory_before = torch.cuda.memory_allocated(device)
    model, tokenizer = get_accelerate_model(args, None)
    model.config.use_cache = False 
    print('loaded model')
    set_seed(args.seed)
    memory_after = torch.cuda.memory_allocated(device)
    memory = (memory_after - memory_before) / (1024**2)
    print(f"model: {memory} MB")

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    syn_lr = torch.tensor(args.lr_teacher).to(device)
    syn_lr = syn_lr.detach().to(device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    wshape = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name: 
            wshape = param.shape
            break

    memory_before = torch.cuda.memory_allocated(device)

    sentence_length = random.randint(32, 1024)
    syn_embeds = torch.randn(args.batch_syn, sentence_length, wshape[1]).to(device).requires_grad_(True)
    attention_mask = create_mask(args.batch_syn, sentence_length)
    syn_labels = creat_labels(args.batch_syn, sentence_length)
    mask = (syn_labels == -100).float()
    syn_labels[syn_labels == -100] = 0
    one_hot = F.one_hot(syn_labels, num_classes=32001)
    syn_labels = F.gumbel_softmax(one_hot.float(), tau=1.0, hard=False, dim=-1).to(device).requires_grad_(True)

    memory_after = torch.cuda.memory_allocated(device)
    memory = (memory_after - memory_before) / (1024**2)
    print(f"initialize distilled dataset: {memory} MB")
    # print("syn_embeds: ", syn_embeds)
    # print("attention mask: ", attention_mask)
    # print("syn_labels: ", syn_labels)

    optimizer_embeds = torch.optim.SGD([syn_embeds], lr=args.lr_text, momentum=0.5)
    optimizer_labels = torch.optim.SGD([syn_labels], lr=args.lr_label, momentum=0.5)
        
    input_dict = {"inputs_embeds": syn_embeds.to(device), "attention_mask": attention_mask.to(device), "labels": syn_labels.to(device)}
    # # get the inputs for trainer
    # dataloader = trainer.get_train_dataloader(batch_size=2)
    # keys = []
    # for batch in dataloader:
    #     input = trainer._prepare_inputs(batch)
    #     keys = [k for k in input]
    #     break

    # syn_text = [[] for _ in range(len(keys))]
    # masks = []
    # for batch in dataloader:
    #     input = trainer._prepare_inputs(batch)
    #     for k, v in input.items():
    #         if k == 'input_ids':
    #             one_hot_v = F.one_hot(v, num_classes=32001)
    #             v = F.gumbel_softmax(one_hot_v.float(), tau=1.0, hard=False, dim=-1).requires_grad_(True)
    #             syn_text[0].append(v)
    #         elif k == 'labels':
    #             mask = (v == -100).float()
    #             masks.append(mask)
    #             v[v == -100] = 0
    #             one_hot_v = F.one_hot(v, num_classes=32001)
    #             gumbel_v = F.gumbel_softmax(one_hot_v.float(), tau=1.0, hard=False, dim=-1)
    #             syn_text[2].append(gumbel_v.requires_grad_(True))
    #         else:
    #             for i, key in enumerate(keys):
    #                 if k == key:
    #                     syn_text[i].append(v.float().requires_grad_(True))
    #                     break

    # # define the optimizers
    # optimizer_list = [None] * (len(keys))
    # for i in range(len(optimizer_list)):
    #     optimizer_list[i] = torch.optim.SGD(syn_text[i], lr=args.lr_text, momentum=0.5)
    # optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()[1:]


    # if args.load_all:
    #     buffer = []
    #     n = 0
    #     while os.path.exists(os.path.join(args.expert_dir, "replay_buffer_{}.pt".format(n))):
    #         buffer = buffer + torch.load(os.path.join(args.expert_dir, "replay_buffer_{}.pt".format(n)), map_location='cuda:0')
    #         n += 1
    #     if n == 0:
    #         raise AssertionError("No buffers detected at {}".format(args.expert_dir))
    # else:
    #     expert_files = []
    #     n = 0
    #     while os.path.exists(os.path.join(args.expert_dir, "replay_buffer_{}.pt".format(n))):
    #         expert_files.append(os.path.join(args.expert_dir, "replay_buffer_{}.pt".format(n)))
    #         n += 1
    #     if n == 0:
    #         raise AssertionError("No buffers detected at {}".format(args.expert_dir))
    #     file_idx = 0
    #     expert_idx = 0
    #     random.shuffle(expert_files)
    #     if args.max_files is not None:
    #         expert_files = expert_files[:args.max_files]
    #     print("loading file {}".format(expert_files[file_idx]))
    #     buffer = torch.load(expert_files[file_idx], map_location='cuda:0')
    #     if args.max_experts is not None:
    #         buffer = buffer[:args.max_experts]
    #     random.shuffle(buffer)

    memory_before = torch.cuda.memory_allocated(device)
    x = torch.load(os.path.join(args.expert_dir, "replay_buffer_0.pt"))
    target_params = torch.cat([v.data.to(device).reshape(-1) for k, v in x[0][-1].items() if 'lora_' in k or 'bias' in k], 0)
    del x
    memory_after = torch.cuda.memory_allocated(device)
    memory = (memory_after - memory_before) / (1024**2)
    print(f"load trajectory: {memory} MB")

    for it in range(0, args.Iteration+1):
        # wandb.log({"Progress": it}, step=it)
        save_this_it = False
        
        # expert_trajectory = buffer
        # if args.load_all:
        #     expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        # else:
        #     expert_trajectory = buffer[expert_idx]
        #     expert_idx += 1
        #     if expert_idx == len(buffer):
        #         expert_idx = 0
        #         file_idx += 1
        #         if file_idx == len(expert_files):
        #             file_idx = 0
        #             random.shuffle(expert_files)
        #         print("loading file {}".format(expert_files[file_idx]))
        #         if args.max_files != 1:
        #             del buffer
        #             buffer = torch.load(expert_files[file_idx], map_location='cuda:0')
        #         if args.max_experts is not None:
        #             buffer = buffer[:args.max_experts]
        #         random.shuffle(buffer)
        
        # # get dataset for evaluation or saving
        # if (it in eval_it_pool) or save_this_it or (it % 1000 == 0):
        #     batches = []
        #     for i in range(len(syn_text)):
        #         batch = {}
        #         for j in range(len(keys)):
        #             batch.update({keys[j]: torch.round(syn_text[i][:,:,j]).detach().long()})
        #         batches.append(batch)
            
        #     dataset = CustomDataset(batches)
        #     metadata = {
        #         'features': ['input_ids', 'attention_mask', 'labels'],
        #         'num_rows': len(dataset)
        #     }
        #     eval_dataset = {'Dataset': dataset, 'metadata': metadata}

        # if it in eval_it_pool:
        #     lr = syn_lr.item()

        #     # get model for evaluation
        #     eval_model, gradient_accumulation_steps, ddp = get_model(base_model=base_model, batch_size=batch_size, micro_batch_size=micro_batch_size,
        #                                                 lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)

        #     eval_trainer = transformers.Trainer(
        #         model=eval_model,
        #         train_dataset=eval_dataset,
        #         eval_dataset=val_data,
        #         compute_metrics=compute_metrics,
        #         args=transformers.TrainingArguments(
        #             per_device_train_batch_size=micro_batch_size,
        #             gradient_accumulation_steps=gradient_accumulation_steps,
        #             warmup_steps=100,
        #             num_train_epochs=epoch_eval_train,
        #             learning_rate=float(lr),
        #             fp16=True,
        #             logging_steps=10,
        #             optim="adamw_torch",
        #             evaluation_strategy="steps" if val_set_size > 0 else "no",
        #             save_strategy="steps",
        #             eval_steps=200 if val_set_size > 0 else None,
        #             save_steps=200,
        #             output_dir=output_dir,
        #             save_total_limit=3,
        #             load_best_model_at_end=True if val_set_size > 0 else False,
        #             ddp_find_unused_parameters=False if ddp else None,
        #             group_by_length=group_by_length,
        #         ),
        #         data_collator=transformers.DataCollatorForSeq2Seq(
        #             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #         ),
        #         callbacks=[SavePeftModelCallback],
        #     )
        #     eval_model.config.use_cache = False

        #     old_state_dict = eval_model.state_dict
        #     eval_model.state_dict = (
        #         lambda self, *_, **__: get_peft_model_state_dict(
        #             self, old_state_dict()
        #         )
        #     ).__get__(eval_model, type(eval_model))

        #     if torch.__version__ >= "2" and sys.platform != "win32":
        #         eval_model = torch.compile(eval_model)

        #     eval_trainer.train()
        #     metric = eval_trainer.evaluate()

        #     print("metric: ", metric)

        #     if (metric['eval_bleu'] > best_metric):
        #         best_metric = metric
        #         save_this_it = True
            
        #     # wandb.log({'BLEU': metric['eval_bleu']}, step=it)

        # if (save_this_it or it % 1000 == 0):
        #     with torch.no_grad():
        #         text_save = eval_dataset
        #         save_dir = os.path.join(".", "logged_files")

        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)

        #         torch.save(text_save, os.path.join(save_dir, "text_{}.json".format(it)))

        #         if save_this_it:
        #             torch.save(text_save, os.path.join(save_dir, "text_best.json"))
        #             torch.save(syn_lr.item(), os.path.join(save_dir, "lr_best.pt"))


        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)
        
        # target_params = expert_trajectory[-1]
        # target_params = torch.cat([v.data.to(device).reshape(-1) for k, v in target_params.items() if 'lora_' in k or 'bias' in k], 0)

        # starting_params = torch.cat([v.data.to(device).reshape(-1) for k, v in model.named_parameters() if 'lora_' in k or 'bias' in k], 0)
        student_params = [torch.cat([v.data.to(device).reshape(-1) for k, v in model.named_parameters() if 'lora_' in k or 'bias' in k], 0).requires_grad_(True)]

        prepared_inputs = trainer._prepare_inputs(input_dict)
        model.to(device)
        model = ReparamModule(model)

        # if distributed:
        #     for i in model.named_parameters():
        #         print(f"{i[0]} -> {i[1].device}")
        #     for k, v in prepared_inputs.items():
        #         print(f"inputs device -> {v.device}")
        #     model = torch.nn.DataParallel(model)

        model.zero_grad()

        for epoch in range(0, args.syn_steps):

            # index = random.randint(0, len(syn_text[0])-1)
            # input_dict = {}
            # for i in range(len(keys)):
            #     input_dict[keys[i]] = syn_text[i][index]

            model.train()

            # prepared_inputs = trainer._prepare_inputs(input_dict)

            # if distributed:
            #     forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            # else:
            forward_params = student_params[-1]

            with trainer.compute_loss_context_manager():
                memory_before = torch.cuda.memory_allocated(device)
                loss = trainer.compute_loss(model, prepared_inputs, flat_param=forward_params)
                memory_after = torch.cuda.memory_allocated(device)
                memory = (memory_after - memory_before) / (1024**2)
                print(f"train: {memory} MB")
                # mask = masks[index]
                # loss = (loss * (1 - mask)).sum() / (1 - mask).sum()
                print("loss: ", loss)

            print("begin backward")
            memory_before = torch.cuda.memory_allocated(device)
            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
            # loss.backward(retain_graph=True)
            mem_after_backward = torch.cuda.memory_allocated()
            mem_allocated_during_backward = (mem_after_backward - mem_before_backward) / (1024 **2)
            print(f"Memory allocated during Backward: {mem_allocated_during_backward} MB")

            # print("grad input: ", syn_embeds.grad)
            # grad = forward_params.grad
            print("grad: ", grad)

            tmp = student_params[-1] - syn_lr * grad
            student_params.append(tmp)
        
        # wandb.log({"Train_Loss": loss.detach().cpu()})

        # compute parameter distance loss and update distilled dataset
        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(student_params[0], target_params, reduction="sum")
        print("iter = %04d, param_loss = %.4f, param_dist = %.4f" % (it, param_loss, param_dist))
        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_embeds.zero_grad()
        optimizer_labels.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        # Print out the gradients
        print("Gradient of embedding: ", syn_embeds.grad)
        pritn("Gradient of labels: ", syn_labels.grad)
        print("Gradient of syn_lr:", syn_lr.grad)

        optimizer_embeds.step()
        optimizer_labels.step()
        optimizer_lr.step()

        # wandb.log({"Grand_Loss": grand_loss.detach().cpu()})

        for _ in student_params:
            del _

        model = retrieve(model)

        if it%10 == 0:
            print('iter = %04d, loss = %.4f' % (it, grand_loss.item()))

    # wandb.finish()

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        timestamps = []
        trajectories = []
        train_result, timestamps = trainer.train(timestamps=timestamps)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

        print("timestamps length: ", len(timestamps))

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        trajectories.append(timestamps)
        if len(trajectories) == args.save_interval:
                n = 0
                while os.path.exists(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n))):
                    n += 1
                print("Saving {}".format(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n))))
                torch.save(trajectories, os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n)))
                trajectories = []
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    distill()
