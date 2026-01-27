import torch
from transformers import AutoTokenizer
from model import MultilingualHateModel # <--- Importing your class from File 1

# CONFIGURATION
MODEL_PATH = 'best_multilingual_model.pt' # Make sure this file is in the same folder
MODEL_NAME = 'xlm-roberta-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🔄 Loading System on {device.upper()}...")

# 1. LOAD TOKENIZER & ARCHITECTURE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MultilingualHateModel(MODEL_NAME)

# 2. LOAD WEIGHTS (Handling the Multi-GPU 'module.' prefix)
state_dict = torch.load(MODEL_PATH, map_location=device)
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k.replace("module.", "") # Cleaning the keys
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval() # Important: Disables Dropout for stable predictions
print("✅ Model Successfully Loaded!")

# 3. DEFINE PREDICTION FUNCTION (API Endpoint Logic)
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
        
        # Threshold Tuning: Only flag if > 75% confident
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
        "confidence_score": round(hate_prob * 100, 2)
    }

# 4. TEST BLOCK (Runs only if you execute this file directly)
if __name__ == "__main__":
    test_text = input("Enter a sentence to test: ")
    result = get_prediction(test_text)
    print(f"\n📊 RESULT: {result}")