from unsloth import FastLanguageModel
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Load your dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def format(x):
    return {"text": x["text"]}
dataset = dataset.map(format)

# Train
model.fit(
    dataset=dataset,
    tokenizer=tokenizer,
    output_dir="./llama3-finetuned",
    max_steps=1000,        # adjust based on dataset
    batch_size=2,
    lr=2e-4,
)

model.save_pretrained("./llama3-finetuned")
tokenizer.save_pretrained("./llama3-finetuned")
