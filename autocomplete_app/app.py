# app.py
from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

model_name = "./aragpt2-finetuned/checkpoint-12500"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    input_text = data.get("input_text", "")
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 10,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggested_part = generated_text[len(input_text):].strip().split()
    suggestions = list(dict.fromkeys(suggested_part))[:5]

    return jsonify({"suggestions": suggestions})

if __name__ == '__main__':
    app.run(debug=True)
