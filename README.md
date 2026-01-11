# Malayalam-to-English-translator
Translation of a low resource language using mBART
This project fine-tunes Facebook's mBART-large-50-many-to-many-mmt model for translating Malayalam text to English using a Bible dataset.

## Overview

The notebook `LLM final PHASE2.ipynb` contains the complete pipeline for:
- Loading and preprocessing the Bible dataset (Malayalam-English pairs)
- Tokenizing the data using mBART tokenizer
- Fine-tuning the mBART model for translation
- Evaluating the model using BLEU, ROUGE, and METEOR metrics
- Translating new Malayalam sentences to English

## Dataset

The dataset used is `bible_50.csv`, which contains parallel Malayalam and English text from the Bible. It includes columns:
- `malayalam_text`: Source text in Malayalam
- `english_text`: Target text in English

The dataset is split into 80% training and 20% testing.

## Requirements

- Python 3.8+
- Libraries: transformers, datasets, evaluate, pandas, numpy
- GPU recommended for training (set `fp16=True` in training arguments if available)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets evaluate pandas numpy
   ```

3. Ensure you have the dataset file `bible_50.csv` in the project directory.

## Usage

1. Open the Jupyter notebook `LLM final PHASE2.ipynb`.

2. Run the cells in order to:
   - Load and preprocess the data
   - Load the tokenizer and model
   - Tokenize the dataset
   - Set up training arguments and trainer
   - Train the model
   - Evaluate or translate new text

### Training

The model is trained for 5 epochs with:
- Learning rate: 3e-5
- Batch size: 4
- Evaluation strategy: per epoch

### Evaluation

Metrics computed:
- BLEU score
- METEOR score
- ROUGE-L score
- Average generation length

### Translation Example

To translate a Malayalam sentence:
```python
input_text = "ഇവൻ എന്റെ സഹോദരൻ ആണ്"
inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["en_XX"]
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Model Details

- **Model**: facebook/mbart-large-50-many-to-many-mmt
- **Source Language**: Malayalam (ml_IN)
- **Target Language**: English (en_XX)
- **Max Length**: 128 tokens

## Results

After training, the model can be used for Malayalam to English translation. Evaluate the performance on the test set using the provided metrics.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
