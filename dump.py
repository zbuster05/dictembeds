








# ---------------------------------------------
# context = r"""
# Quantum computing is the use of quantum phenomena such as superposition and entanglement to perform computation. Computers that perform quantum computations are known as quantum computers.[1]:I-5 They are believed to be able to solve certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers. The study of quantum computing is a subfield of quantum information science. It is likely to expand in the next few years as the field shifts toward real-world use in pharmaceutical, data security and other applications. [2]
# """

# questions = [
        # "What problems could quantum computers solve?",
# ]

# for question in questions:
    # # encode input and context
    # inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", padding=True)

    # # do maff
    # model_result = model(**inputs)

    # # get the start and end location logits
    # start_logits = model_result["start_logits"]
    # end_logits = model_result["end_logits"]

    # # get the top k=1 likely start and end locations
    # answer_start = torch.argmax(start_logits) 
    # answer_end = torch.argmax(end_logits) + 1 

    # # get the input token IDs as a list
    # input_ids = inputs["input_ids"].tolist()[0]
    # # and slice for our desired start and end
    # answer_ids = input_ids[answer_start:answer_end]

    # # unvectorize
    # answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    # # untokenize
    # answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # breakpoint()

