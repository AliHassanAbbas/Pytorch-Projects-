# Sentiment Analysis with PyTorch

A progressive set of hands-on notebooks for binary sentiment classification, moving from simple word representations to recurrent, convolutional, and Transformer-based models.

The notebooks use the **IMDb movie-review dataset** and build a common text-classification pipeline: tokenization, vocabulary or tokenizer IDs, padding, batching, model training, validation, and evaluation.

## Notebook roadmap

| Notebook | Approach | Main concepts |
|---|---|---|
| [0 — Neural Bag of Words (NBoW)](0_Sentiment_Analysis_with_NBoW_or_CBoW.ipynb) | Embedding-based bag-of-words classifier | Tokenization, vocabulary construction, numericalization, padding, embedding lookup, GloVe initialization |
| [1 — LSTM](1_Sentiment_Analysis_with_LSTM_in_Pytorch.ipynb) | Recurrent sequence classifier | Sequence lengths, packed padded sequences, hidden and cell states, bidirectional LSTM, dropout, pretrained embeddings |
| [2 — CNN](2_Sentiment_Analysis_With_CNN_in_Pytorch.ipynb) | 1D convolutional text classifier | Word embeddings, tensor dimension permutation, Conv1D filters, feature extraction, pooling/classification |
| [3 — Transformer / BERT](3_Sentiment_analysis_with_Transformers_in_pytorch_.ipynb) | BERT-based classifier | Hugging Face tokenizer/model loading, pretrained contextual representations, attention outputs, fine-tuning or freezing the encoder |

## What you will learn

### Text preprocessing and data pipelines
- Load the IMDb dataset with Hugging Face Datasets.
- Tokenize reviews with TorchText's basic English tokenizer or a pretrained BERT tokenizer.
- Build a vocabulary with unknown and padding tokens, then convert tokens to integer IDs.
- Split training data into training and validation subsets.
- Pad variable-length sequences and create mini-batches with a custom collate function and PyTorch DataLoader.

### Text representation
- Learnable word embeddings with nn.Embedding.
- Initialize embedding weights with pretrained GloVe vectors.
- Understand the difference between a bag-of-words representation and sequence-aware representations.
- Use BERT's pretrained subword tokenizer and contextual token representations.

### Neural architectures
- **NBoW:** aggregate word embeddings to represent a review before classification.
- **LSTM:** model word order and sequence context; handle variable-length batches with pack_padded_sequence.
- **CNN:** apply nn.Conv1d filters over embedded token sequences to extract local text features.
- **BERT:** use a pretrained Transformer encoder through Hugging Face Transformers and attach a classification head.

### Training and evaluation
- Define models with torch.nn.Module.
- Train with CrossEntropyLoss and Adam.
- Implement training and evaluation loops with model.train() and model.eval().
- Track loss and accuracy across training and validation.
- Save the best-performing model checkpoint using validation loss.
- Plot learning curves and test models on new text where implemented.

## Technical stack

- Python
- PyTorch (torch, torch.nn, torch.optim)
- TorchText (tokenization, vocabulary utilities, GloVe vectors)
- Hugging Face Datasets (IMDb data loading and preprocessing)
- Hugging Face Transformers (BERT tokenizer and pretrained encoder)
- NumPy, Matplotlib, and tqdm

## Run the notebooks

These notebooks were authored for a Google Colab/Jupyter workflow. Open a notebook and run its cells in order.

The original notebooks pin older versions of PyTorch and TorchText (including PyTorch 2.3.0 and TorchText 0.18.0). For reproducibility, use a compatible environment rather than assuming the latest package versions will work unchanged.

Typical dependencies used across the notebooks:

```bash
pip install torch==2.3.0 torchtext==0.18.0 datasets transformers matplotlib numpy tqdm
```

Some notebooks download IMDb data and GloVe vectors, while the Transformer notebook downloads the bert-base-uncased tokenizer and pretrained model. Internet access is therefore required for initial setup.

## Suggested learning order

1. Start with NBoW to understand token IDs, vocabulary, embeddings, and batching.
2. Move to LSTM to learn sequence modelling and packed sequences.
3. Study CNN to understand local n-gram feature extraction with convolutions.
4. Finish with BERT to see how pretrained Transformer representations are used for classification.

## Scope and implementation note

This is a learning-oriented collection of sentiment-analysis implementations, not a benchmark comparison. The BERT notebook loads a pretrained bert-base-uncased model; it does **not** implement or pretrain a Transformer language model from scratch. Check each notebook's runtime outputs and configuration before comparing accuracy or training cost.

## Author

**Ali Hassan Abbas** — [GitHub](https://github.com/AliHassanAbbas)

Explore the complete [PyTorch Projects repository](https://github.com/AliHassanAbbas/Pytorch-Projects-).
