import random
import numpy as np
import torch


SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import unsloth
from datasets import load_dataset
import weave
import asyncio
from unsloth import FastLanguageModel


max_seq_length = 2048
dtype = None
load_in_4bit = False
BASE_MODEL_NAME = "unsloth/Qwen3-0.6B"
LORA_MODEL_DIR = "lora_model"
N = 30


weave.init("q3")


# === GLOBAL: LOAD MODELS ONLY ONCE ===
BASE_MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
LORA_MODEL, _ = FastLanguageModel.from_pretrained(
    model_name=LORA_MODEL_DIR,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
BASE_MODEL.eval().to("cuda")
LORA_MODEL.eval().to("cuda")
FastLanguageModel.for_inference(BASE_MODEL)
FastLanguageModel.for_inference(LORA_MODEL)


def make_prompt(instruction, input_text):
    if input_text.strip():
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction
    return [{"role": "user", "content": user_message}]


def apply_chat_template_loss(sample, tokenizer):
    messages = make_prompt(sample["instruction"], sample["input"])
    messages.append({"role": "assistant", "content": sample["output"]})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def apply_chat_template_generation(sample, tokenizer):
    messages = make_prompt(sample["instruction"], sample["input"])
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def output_only_loss(tokenizer, model, sample, device="cuda"):
    # 1. Prepare full prompt+output for loss
    prompt_plus_output = apply_chat_template_loss(sample, tokenizer)
    # 2. Prepare prompt only (for prefix length)
    prompt_only = make_prompt(sample["instruction"], sample["input"])
    prompt_only_str = tokenizer.apply_chat_template(
        prompt_only,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # 3. Tokenize both
    tok_full = tokenizer(
        prompt_plus_output,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"  # For safe shape ops
    )
    tok_prompt = tokenizer(
        prompt_only_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length
    )
    input_ids = tok_full["input_ids"].to(device)
    labels = input_ids.clone()


    # 4. Loss ONLY on output tokens
    prompt_len = tok_prompt["input_ids"].shape[-1]   # prompt tokens count (may be == 2048!)
    # Mask prompt tokens in labels
    labels[:, :prompt_len] = -100
    # Mask pad tokens if there
    if tokenizer.pad_token_id is not None:
        labels[input_ids == tokenizer.pad_token_id] = -100


    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss.item()
    return loss


def safe_generate(model, tokenizer, prompt, device="cuda"):
    # Tokenize prompt and ensure we never overflow model max length
    prompt_tok = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length
    ).to(device)
    prompt_len = prompt_tok['input_ids'].shape[1]
    # Prevent overflow: at least generate 1, never beyond 2048
    max_gen = max(1, max_seq_length - prompt_len)
    with torch.no_grad():
        output = model.generate(
            **prompt_tok,
            max_new_tokens=max_gen,
            use_cache=False,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )
        out_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return out_text


class QwenBaseModel(weave.Model):
    @weave.op()
    async def predict(self, instruction, input, output):
        sample = {
            "instruction": instruction,
            "input": input,
            "output": output,
        }
        # LOSS on output tokens only
        loss = output_only_loss(TOKENIZER, BASE_MODEL, sample)
        # GENERATION safely
        prompt_gen = apply_chat_template_generation(sample, TOKENIZER)
        output_text = safe_generate(BASE_MODEL, TOKENIZER, prompt_gen)
        return {"loss": loss, "output": output_text}


class QwenLoraModel(weave.Model):
    @weave.op()
    async def predict(self, instruction, input, output):
        sample = {
            "instruction": instruction,
            "input": input,
            "output": output,
        }
        # LOSS on output tokens only
        loss = output_only_loss(TOKENIZER, LORA_MODEL, sample)
        # GENERATION safely
        prompt_gen = apply_chat_template_generation(sample, TOKENIZER)
        output_text = safe_generate(LORA_MODEL, TOKENIZER, prompt_gen)
        return {"loss": loss, "output": output_text}


@weave.op()
def loss_only_scorer(output):
    return {"loss": output["loss"]}


# ====== Load LAST 10% of train and pick 30 samples ======
full_ds = load_dataset("yahma/alpaca-cleaned", split="train")
length = len(full_ds)
start = int(length * 0.9)
end = length
ds_last10 = full_ds.select(range(start, end))
samples = [
    dict(
        instruction=row["instruction"],
        input=row["input"],
        output=row["output"]
    )
    for row in ds_last10.select(range(N))
]


async def main():
    models = {
        "Qwen3-0.6B-base": QwenBaseModel(),
        "Qwen3-0.6B-LoRA": QwenLoraModel(),
    }
    scorers = [loss_only_scorer]


    for model_name, model in models.items():
        print(f"\n=== Evaluating {model_name} ===")
        evaluation = weave.Evaluation(
            dataset=samples,
            scorers=scorers,
            name=f"{model_name} LossEval"
        )
        results = await evaluation.evaluate(model)


if __name__ == "__main__":
    asyncio.run(main())
