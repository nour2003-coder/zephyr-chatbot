from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

model_id = "nouri20000/employee-chatbot-fr-merged"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
