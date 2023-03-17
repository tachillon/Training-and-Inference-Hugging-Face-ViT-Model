from datasets import load_dataset
from transformers import ViTImageProcessor
import torch
import numpy as np
from datasets import load_metric
# from evalutate import load
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer

# Load images from a local folder
# 1. resize to 224x224 (ViT requires images to be at least 224x224) the dataset folder with norm.py
# 2. split the dataset into train, validation and test with split_dataset.py
data_dir = "dataset_split"
ds = load_dataset(
    "imagefolder", data_dir=data_dir, cache_dir="/tmp/cache", drop_labels=False
)

# Load the ViT model
model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["label"] = example_batch["label"]
    return inputs

prepared_ds = ds.with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )

labels = ds["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

training_args = TrainingArguments(
  output_dir="./vit-base-custom-dataset",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds["validation"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "image-classification",
    "dataset": "beans",
    "tags": ["image-classification"],
}

if training_args.push_to_hub:
    trainer.push_to_hub("🍻 cheers", **kwargs)
else:
    trainer.create_model_card(**kwargs)
