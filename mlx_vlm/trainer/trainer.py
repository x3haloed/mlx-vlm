import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

# NOTE: This trainer supports trajectory-aware sleep training.
# - Standard SFT batches may include example_weight derived from future outcome g(O_t)
# - Preference batches use (preferred_input_ids, rejected_input_ids) for DPO-style loss


def get_prompt(model_type, processor, conversation):
    if model_type == "paligemma":
        return conversation

    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif "tokenizer" in processor.__dict__.keys():
        prompt = processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    return prompt


class Dataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
    ):
        if split is not None:
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs

        item = self.dataset[idx]
        batch_items = item if isinstance(item, list) else [item]

        images = []
        conversations = []
        for entry in batch_items:
            images.append(entry.get("images", entry.get("image", None)))
            conv = entry.get("messages", entry.get("conversations"))
            if conv is None:
                conv = []
            if isinstance(conv, str):
                conv = [{"role": "user", "content": conv}]
            elif isinstance(conv, dict):
                conv = [conv]
            conversations.append(conv)
        if len(images) == 1:
            images = images[0]
        if images in (None, "", []):
            images = None
        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(images, list):
            empty = True
            for item in images:
                if item is None:
                    continue
                if isinstance(item, str) and item == "":
                    continue
                if isinstance(item, list) and len(item) == 0:
                    continue
                empty = False
                break
            if empty:
                images = None
        prompts = []

        for idx_offset, conversation in enumerate(conversations):
            if not conversation:
                warnings.warn(f"Skipping empty conversation at batch index {idx_offset}")
                continue
            if self.config["model_type"] == "pixtral":
                conversation = [json.loads(i) for i in conversation]
                if len(conversations) > 1:
                    warnings.warn(
                        "Pixtral batch processing is not supported yet. Set batch size to 1."
                    )
            prompt = get_prompt(
                self.config["model_type"], self.processor, conversation
            )
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt, ensure_ascii=False)
            prompts.append(prompt)
        if not prompts:
            warnings.warn("All conversations in batch were empty; skipping batch.")
            return {
                "input_ids": mx.array([], dtype=mx.int32),
                "attention_mask": mx.array([], dtype=mx.int32),
                "pixel_values": None,
            }

        image_token_index = self.config["image_token_index"]

        inputs = prepare_inputs(
            processor=self.processor,
            images=images,
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        if "input_ids" in inputs and inputs["input_ids"].size == 0:
            warnings.warn("Empty input_ids after tokenization; skipping batch.")
            return {
                "input_ids": mx.array([], dtype=mx.int32),
                "attention_mask": mx.array([], dtype=mx.int32),
                "pixel_values": None,
            }
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        mask = inputs["attention_mask"]
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        if mask is None:
            mask = mx.ones_like(input_ids)

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
            **kwargs,
        }
        example_weight = None
        if isinstance(item, list):
            weights = []
            for entry in batch_items:
                weights.append(entry.get("example_weight", entry.get("sample_weight", entry.get("weight"))))
            example_weight = weights
        else:
            example_weight = item.get("example_weight", item.get("sample_weight", item.get("weight")))
        if example_weight is not None:
            if isinstance(example_weight, list):
                output["example_weight"] = mx.array(example_weight)
            else:
                output["example_weight"] = mx.array([float(example_weight)])
        # Optional preference learning support (trajectory-aware sleep)
        # Expect keys: preferred_input_ids / rejected_input_ids (already tokenized)
        if "preferred_input_ids" in item and "rejected_input_ids" in item:
            output["preferred_input_ids"] = mx.array(item["preferred_input_ids"])
            output["rejected_input_ids"] = mx.array(item["rejected_input_ids"])
        return output


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )


def default_loss(model, inputs, targets, lengths):
    logits = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_on_completions=False,
        assistant_id=77091,
        clip_gradients=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_on_completions = train_on_completions
        self.assistant_id = assistant_id
        self.clip_gradients = clip_gradients

    def preference_loss(self, model, batch):
        """DPO-style preference loss for trajectory-aware sleep."""
        pref_ids = batch["preferred_input_ids"]
        rej_ids = batch["rejected_input_ids"]

        def score(input_ids):
            logits = model(input_ids).logits.astype(mx.float32)
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            ce = nn.losses.cross_entropy(logits, labels)
            return -ce.mean(axis=1)  # higher is better

        s_pref = score(pref_ids)
        s_rej = score(rej_ids)

        # DPO-style objective: maximize preference margin
        return -mx.mean(mx.log(mx.sigmoid(s_pref - s_rej)))

    def loss_fn(self, model, batch):
        # Trajectory-aware preference batch
        if "preferred_input_ids" in batch and "rejected_input_ids" in batch:
            return self.preference_loss(model, batch)
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids[:, 1:]

        batch_size, seq_length = input_ids.shape

        if self.train_on_completions:
            weight_mask = mx.ones_like(attention_mask)

            assistant_response_index = np.where(input_ids == self.assistant_id)[1]
            range_matrix = mx.repeat(
                mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
            )
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
                -1, 1
            )
            # Apply the mask to weight_mask
            weight_mask = mx.where(
                assistant_mask, mx.zeros_like(weight_mask), weight_mask
            )[:, 1:]
        else:
            weight_mask = None

        input_ids = input_ids[:, :-1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]
        lengths = mx.sum(attention_mask, axis=1)
        # example_weight is assumed to already encode future-outcome quality g(O_t)
        example_weight = batch.get("example_weight")
        kwargs = {
            k: v
            for k, v in batch.items()
            if k not in ["input_ids", "pixel_values", "attention_mask", "example_weight"]
        }

        # Forward pass
        outputs = model(input_ids, pixel_values, attention_mask, **kwargs)

        # Cast to float32
        logits = outputs.logits.astype(mx.float32)

        # Ensure logits and labels have the same sequence length
        def align_logits_with_labels(logits, labels):
            if logits.shape[1] < labels.shape[1]:
                pad_length = labels.shape[1] - logits.shape[1]
                pad_width = ((0, 0), (0, pad_length), (0, 0))
                return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
            elif logits.shape[1] > labels.shape[1]:
                return logits[:, -labels.shape[1] :, :]
            return logits

        logits = align_logits_with_labels(logits, labels)

        length_mask = mx.arange(input_ids.shape[1])[None, :] < lengths[:, None]

        # Compute loss only on non-padded tokens
        ce = (
            nn.losses.cross_entropy(
                logits,
                labels,
                weights=weight_mask,
            )
            * length_mask
        )
        if example_weight is not None:
            if example_weight.shape[0] != ce.shape[0]:
                print(
                    f"[debug] example_weight shape={example_weight.shape} batch={ce.shape[0]}"
                )
                example_weight = example_weight[: ce.shape[0]]
            if len(example_weight.shape) > 1:
                example_weight = mx.squeeze(example_weight)
            example_weight = mx.expand_dims(example_weight, axis=1)
            ce = ce * example_weight
            denom = (length_mask * example_weight).sum()
        else:
            denom = length_mask.sum()
        ce = ce.sum() / mx.maximum(denom, 1)

        return ce

    def train_step(self, batch):
        if batch.get("input_ids") is not None and batch["input_ids"].size == 0:
            return mx.array(0.0)
        # Optional safety: prevent mixing preference + SFT in same batch
        if "preferred_input_ids" in batch and "input_ids" in batch:
            raise ValueError("Batch must be either SFT or preference, not both")
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)

        # Add gradient clipping
        if self.clip_gradients is not None:
            grads = tree_map(
                lambda g: mx.clip(g, -self.clip_gradients, self.clip_gradients), grads
            )

        self.optimizer.update(self.model, grads)

        return loss

    @mx.compile
    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            mx.eval(self.model, self.optimizer.state)
            total_loss += loss
        return total_loss / len(dataloader)


def save_adapter(
    model: nn.Module,
    adapter_file: Union[str, Path],
):
    path = Path(adapter_file)
    if hasattr(model.config, "lora"):
        with open(path.parent / "adapter_config.json", "w") as f:
            json.dump(model.config.lora, f)
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))
