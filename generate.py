# use_saved_model.py
from model_utils import load_model, generate_text

model_load_path = "bigram_language_model.pt"
loaded_model = load_model(model_load_path)

max_new_tokens = 500
generated_text = generate_text(loaded_model, max_new_tokens, "Test test 123")
print(generated_text)