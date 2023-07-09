from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask import request

app = Flask(__name__)
Bootstrap(app)
tokenizer = T5Tokenizer.from_pretrained('t5-base')


def summarize_t(text):
    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Tokenize and encode the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", model_max_length=800, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=500, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Calculate the token count
    token_count = len(tokenizer.tokenize(text))
    summary_token_count = len(tokenizer.tokenize(summary))

    return summary, token_count, summary_token_count


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        # using t5 model
        summary_t, token_count, summary_token_count = summarize_t(text)

        return render_template('result.html', text=text, summary=summary_t, count_original=token_count,
                               count_result=summary_token_count)

    return render_template('index.html')

# text = "API is the acronym for application programming interface, a software intermediary that allows two applications to talk to each other. APIs are an accessible way to extract and share data within and across organizations."

# USING BERT MODEL (EXTRACTIVE METHOD)
# def summarize_b(input_text):
#     # Load the pre-trained BERT model for extractive summarization
#     summarizer = pipeline("summarization")
#
#     # Generate the summary
#     summary = summarizer(input_text, max_length=50, min_length=10, do_sample=False)
#
#     return summary[0]['summary_text']

app.run(debug=True)