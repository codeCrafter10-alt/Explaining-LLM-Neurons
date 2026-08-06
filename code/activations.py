import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

from dataset import load_dataset, prepare_data

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_config = PeftConfig.from_pretrained("finetuned_model")

model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=3, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "finetuned_model", torch_dtype=torch.float16, device_map="auto")

model.eval()

activations = []

def save_activations(module, inputs, output):
    activations.append(output.detach().cpu())

hook = model.base_model.model.model.layers[-1].mlp.register_forward_hook(save_activations)


dev_file = "data/snli_1.0/snli_1.0_dev.txt"

df = load_dataset(dev_file)
dataset = prepare_data(df)


for example in dataset:
    inputs = tokenizer(
        example["premise"],
        example["hypothesis"],
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)


hook.remove()

torch.save(activations, "activations.pt")

print("Finished hooking and saving activations.")
print("Number of activation batches:", len(activations))
print("Activation shape:", activations[0].shape)