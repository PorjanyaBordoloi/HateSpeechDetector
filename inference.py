import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model import MultilingualHateModel

# CONFIGURATION
# ✅ Point this to the quantized file
MODEL_PATH = 'quantized_hinglish_model.pt' 
MODEL_NAME = 'xlm-roberta-base'
device = 'cpu' # Quantized models run best on CPU (Fast!)

print(f"🔄 Loading QUANTIZED System on {device.upper()}...")

# 1. LOAD TOKENIZER & SKELETON
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MultilingualHateModel(MODEL_NAME)

# 2. PREPARE SKELETON FOR QUANTIZATION
# We must tell PyTorch: "Prepare this model to accept 8-bit weights"
# This mirrors exactly how you saved it.
model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

# 3. LOAD THE WEIGHTS
# No need for the 'module.' fix logic usually, as quantization cleans it up
print("mb Loading weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Quantized Model Successfully Loaded!")

# 4. PREDICTION FUNCTION (Unchanged)
def get_prediction(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=100,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, mask)
        probs = torch.softmax(outputs, dim=1)
        
        hate_prob = probs[0][1].item()
        if hate_prob > 0.75:
            label = "HATE SPEECH"
            is_toxic = True
        else:
            label = "SAFE"
            is_toxic = False
            
    return {
        "text": text,
        "label": label,
        "is_toxic": is_toxic,
        "hate_probability": round(hate_prob * 100, 2)
    }

# Test Block
if __name__ == "__main__":
    test_text = input("Enter text: ")
    print(get_prediction(test_text))