from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

VALID_LLMS = ["gpt2", "facebook/bart-large-cnn", "google/flan-t5-base"]
TASK_MAPPING = {
    "summarization": "summarization",
    "question-answering": "question-answering",
    "text-generation": "text-generation",
    "要約": "summarization",
    "質問応答": "question-answering",
    "テキスト生成": "text-generation"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_name = request.form['model_name']
        model_description = request.form['model_description']
        print(f"Model Name: {model_name}, Description: {model_description}")
        return "Model submitted successfully!"
    return render_template('index.html')

@app.route('/compare_llms', methods=['GET', 'POST'])
def compare_llms():
    if request.method == 'POST':
        llms = request.form.getlist('llm')
        task = request.form['task']
        input_data = request.form['input_data']
        print(f"LLMs: {llms}, Task: {task}, Input Data: {input_data}")

        invalid_llms = [llm for llm in llms if llm not in VALID_LLMS]
        if invalid_llms:
            return f"Invalid LLMs selected: {', '.join(invalid_llms)}. Please select from: {', '.join(VALID_LLMS)}"

        results = {}
        for llm in llms:
            try:
                actual_task = TASK_MAPPING.get(task, task)
                generator = pipeline(actual_task, model=llm)
                results[llm] = generator(input_data)[0]['generated_text']
            except Exception as e:
                results[llm] = f"Error: {e}"

        return render_template('comparison_results.html', results=results, input_data=input_data)
    return render_template('compare_llms.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=57049)