from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import torch, os

# Config
MODEL_NAME = os.environ.get("BASE_MODEL", "facebook/bart-large-cnn")
MAX_IN, MAX_OUT = 1024, 128
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 3e-5))
BS = int(os.environ.get("BS", 8))
GRAD_ACC = int(os.environ.get("GRAD_ACC", 2))
OUT_DIR = os.environ.get("OUT_DIR", "./results_dgx")
SAVE_DIR = os.environ.get("SAVE_DIR", "./models/fine_tuned_model")

# Data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data, val_data = dataset["train"], dataset["validation"]

# Model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(ex):
    inputs = tokenizer(ex["article"], max_length=MAX_IN, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(ex["highlights"], max_length=MAX_OUT, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

cols = ["article", "highlights", "id"]
train_tok = train_data.map(preprocess, batched=True, remove_columns=cols)
val_tok   = val_data.map(preprocess,   batched=True, remove_columns=cols)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = TrainingArguments(
    output_dir=OUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    predict_with_generate=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved fine-tuned model to {SAVE_DIR}")