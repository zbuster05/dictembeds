# type: ignore
# pylint: disable=no-member

from execute import Engine

import re
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app)

# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a2fc9:B_VAL::0:24900:0.8616854640095484"
model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a5029:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-cf8cd:ROUTINE::0:20000"

e = Engine(model_path=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.json["title"]
        context = request.json["context"]
    except (KeyError, TypeError):
        return jsonify({"code": "bad_request", "response": "Bad request. Missing key(s) title, context.", "payload": ""}), 400

    params = request.json.get("params")

    try: 
        if params:
            result = e.execute(title.strip(), 
                                re.sub(r"\n", "", context.strip()), 
                                num_beams=int(params["num_beams"]), 
                                min_length=int(params["min_length"]), 
                                no_repeat_ngram_size=int(params["no_repeat_ngram_size"]))
        else:
            result = e.execute(title.strip(), re.sub(r"\n", "", context.strip()))
    except ValueError:
        return jsonify({"code": "size_overflow", "response": f"Size overflow. The context string is too long and should be less than {e.tokenizer.model_max_length} tokens.", "payload": e.tokenizer.model_max_length}), 418
    except (KeyError, TypeError):
        return jsonify({"code": "bad_request", "response": "Bad request. Missing few of key(s) num_beams,  min_length or no_repeat_ngram_size in params.", "payload": ""}), 400

    return {"code": "success", "response": result}, 200

if __name__ == "__main__":
    app.run()


