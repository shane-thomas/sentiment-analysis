# Sentiment Analysis with Neural Networks 🤖

This notebook demonstrates a full pipeline for multi-class sentiment analysis using neural networks and DistilBERT embeddings. Given a dataset of text reviews, it preprocesses the data, extracts embeddings using a pre-trained transformer, trains a custom neural network classifier, and evaluates its performance. You can also use the trained model to predict the sentiment of your own custom input!

## Features

- **Data Preprocessing**: Loads and cleans review data, merges summary and review text, and maps sentiment scores to five classes.
- **Label Engineering**: Casts sentiment labels to human-readable categories: *negative*, *somewhat negative*, *neutral*, *somewhat positive*, *positive*.
- **Efficient Tokenization**: Uses DistilBERT's tokenizer for fast, robust text encoding.
- **Embedding Extraction**: Extracts semantic embeddings from DistilBERT for each review, capturing contextual meaning.
- **Custom Neural Network Classifier**: Trains a multi-layer perceptron on top of the embeddings for accurate multi-class sentiment prediction.
- **Training & Validation**: Includes stratified dataset splitting, batch loading, and live training progress with accuracy and loss stats.
- **Evaluation**: Reports both strict and relaxed accuracy (±1 sentiment class) on the test set.
- **Custom Prediction**: Easily predict the sentiment of any input text using the trained pipeline.

## Prerequisites

Before running the notebook, make sure you have the following installed:

- **Python 3.8+**
- **PyTorch**: For deep learning and model training.
- **Transformers**: For DistilBERT and tokenization.
- **datasets**: For easy data manipulation and splits.
- **tqdm**: For progress bars during processing.
- **CUDA** (optional): For faster training if you have a GPU.

Install all dependencies with:
 ```bash
pip install -r requirements.txt
```

## How to Run

1. **Prepare the Data**  
   Place your review dataset as `Reviews.csv` in the working directory. The CSV should have at least `Summary`, `Text`, and `Score` columns.

2. **Open the Notebook**  
   Launch Jupyter Notebook and open the provided notebook file.

3. **Run All Cells**  
   Execute the notebook cells sequentially. The pipeline will:
   - Preprocess and split the data
   - Tokenize and extract embeddings
   - Train the neural network classifier
   - Evaluate performance on the test set

4. **Try Custom Predictions**  
   At the end, you can enter your own text and see the predicted sentiment class!

## How It Works

1. **Data Preparation**  
   - Loads review data from CSV.
   - Merges summary and review content.
   - Converts scores from 1–5 to 0–4 and assigns sentiment labels.

2. **Tokenization & Embedding**  
   - Uses DistilBERT tokenizer to encode text.
   - Extracts the [CLS] token embedding for each review, representing the whole input.

3. **Neural Network Training**  
   - A simple feed-forward neural network is trained on the embeddings.
   - Training and validation accuracy and loss are displayed per epoch.

4. **Evaluation**  
   - Reports standard accuracy and relaxed accuracy (prediction within ±1 class is also counted as correct).
   - Example: If the true label is "neutral" and the model predicts "somewhat positive," it's considered correct under relaxed accuracy.

5. **Custom Input Prediction**  
   - Enter any text and instantly get its predicted sentiment class, mapped to a readable label.
  
## Example Output
```bash
Test Accuracy: 72.82%
Relaxed Test Accuracy (±1): 90.50%
Predicted Sentiment Class: 5
positive
```


## Sentiment Classes

| Label | Description          |
|-------|----------------------|
|   0   | negative             |
|   1   | somewhat negative    |
|   2   | neutral              |
|   3   | somewhat positive    |
|   4   | positive             |

## Why This Approach?

- **Transfer Learning**: By leveraging DistilBERT, the model benefits from rich, pre-trained language representations, making it robust even with limited data.
- **Custom Classifier**: Training a lightweight neural network on top of embeddings allows for fast experimentation and adaptation to new sentiment categories.
- **Transparent Pipeline**: Each step—from raw data to prediction—is clearly separated, making it easy to understand, modify, or extend.

