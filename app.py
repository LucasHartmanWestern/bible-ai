from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import load_model, generate_text

app = Flask(__name__)
CORS(app)

print("############ APP STARTED ############")

model_load_path = "bigram_language_model.pt"
loaded_model = load_model(model_load_path)

print("############ MODEL LOADED ############")

@app.route('/generate', methods=['POST'])
def generate():

    print('Received')
    data = request.get_json()
    print(data)
    max_new_tokens = data['max_new_tokens']
    starting_string = data['starting_string']
    if starting_string == '': starting_string = " "
    generated_text = generate_text(loaded_model, max_new_tokens, starting_string)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')