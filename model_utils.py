import torch
from train import BigramLanguageModel, decode, device

# load the saved model
def load_model(model_load_path):
    loaded_model = BigramLanguageModel()
    loaded_model.load_state_dict(torch.load(model_load_path))
    loaded_model = loaded_model.to(device)
    return loaded_model

# generate some text
def generate_text(loaded_model, max_new_tokens):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(loaded_model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())