import torch
from train import BigramLanguageModel, encode, decode, device

# load the saved model
def load_model(model_load_path):
    device = torch.device('cpu')
    loaded_model = BigramLanguageModel()
    loaded_model.load_state_dict(torch.load('bigram_language_model.pt', map_location=device))
    loaded_model = loaded_model.to(device)
    return loaded_model

# generate some text
def generate_text(loaded_model, max_new_tokens, starting_string='\n'):
    context = torch.tensor([encode(starting_string)], dtype=torch.long, device=device)
    return decode(loaded_model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())