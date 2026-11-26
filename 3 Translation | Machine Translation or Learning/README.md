# Machine Translation with PyTorch

This folder contains three educational implementations of **German-to-English neural machine translation** using the Multi30k dataset. The notebooks progress from a foundational recurrent encoder–decoder to a GRU model and then an attention-based architecture.

The implementations are built with PyTorch modules and explicit training and inference loops. They are useful for understanding how sequence-to-sequence translation systems are assembled, trained, evaluated, and used to generate translations.

## Notebook roadmap

| Notebook | Architecture / paper | Main concepts |
|---|---|---|
| [0 — Sequence-to-Sequence Learning with Neural Networks](0_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb) | LSTM encoder–decoder | Separate encoder and decoder, hidden and cell states, teacher forcing, autoregressive decoding |
| [1 — Learning Phrase Representations using GRU Encoder–Decoder](1_Learning_Phrase_Representations_using_GRU_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb) | GRU encoder–decoder | Phrase representation, context vector, GRU state flow, teacher forcing |
| [2 — Neural Machine Translation by Jointly Learning to Align and Translate](2_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb) | Bidirectional GRU + additive attention | Encoder annotations, alignment scores, attention weights, context vectors, attention visualization |

## 1. LSTM sequence-to-sequence model

The first notebook implements an encoder–decoder model with LSTMs:

- The encoder embeds the source tokens and summarizes the input sequence into hidden and cell states.
- The decoder receives the encoder states and generates target tokens one step at a time.
- Teacher forcing optionally feeds the ground-truth previous target token during training.
- Inference uses autoregressive decoding and stops when the end-of-sequence token is generated (or the output-length limit is reached).

The notebook also includes padding-aware loss, gradient clipping, validation-loss checkpointing, test-set perplexity, and BLEU evaluation.

## 2. GRU encoder–decoder

The second notebook implements the GRU-based architecture associated with *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*.

Compared with the LSTM implementation, the GRU does not maintain a separate cell state. The encoder produces a context representation that initializes the decoder. The decoder combines the current token embedding with the context representation while generating the translation.

Topics covered include:

- GRU encoder and decoder modules
- Learned token embeddings and dropout
- Context-vector conditioning
- Teacher forcing and autoregressive generation
- Gradient clipping and validation checkpointing
- Perplexity and BLEU evaluation

## 3. Attention-based neural machine translation

The third notebook implements the attention-based approach from *Neural Machine Translation by Jointly Learning to Align and Translate*.

Its main components are:

- **Bidirectional GRU encoder:** processes the source sequence in both directions and produces encoder annotations.
- **Additive attention:** scores each encoder annotation against the current decoder state, then normalizes the scores into attention weights.
- **Context vector:** computes a weighted combination of encoder annotations using the attention weights.
- **GRU decoder:** uses the current target embedding and context vector to predict the next target token.

The notebook includes greedy translation, attention-weight collection, attention heatmap visualization, and BLEU evaluation. This makes it possible to inspect which source tokens receive attention while each target token is generated.

## Shared data and preprocessing pipeline

All three notebooks use the [Multi30k dataset](https://huggingface.co/datasets/bentrevett/multi30k) and work with German source sentences and English target sentences.

The preprocessing workflow includes:

1. Loading train, validation, and test splits.
2. Tokenizing German and English with spaCy language models.
3. Lowercasing tokens and adding `<sos>` and `<eos>` markers.
4. Building separate source and target vocabularies with unknown and padding tokens.
5. Converting tokens to integer IDs.
6. Padding variable-length sequences in a custom DataLoader collate function.

## Training and evaluation techniques

The notebooks demonstrate several important sequence-model training practices:

- `nn.Embedding`, `nn.LSTM`, and `nn.GRU`
- Custom `nn.Module` encoder, decoder, attention, and Seq2Seq classes
- Teacher forcing
- `CrossEntropyLoss(ignore_index=pad_index)` to exclude padding tokens from the loss
- Adam optimization and backpropagation
- Gradient clipping
- Training and validation loss tracking
- Saving and restoring the best validation checkpoint
- Perplexity and BLEU evaluation
- Greedy autoregressive decoding
- Attention visualization (attention-based notebook)

## Requirements

The notebooks were authored around the following package versions and dependencies:

- Python
- PyTorch 2.3.0
- TorchText 0.18.0
- Torchvision (installed in the original setup cells)
- Hugging Face Datasets
- Evaluate
- spaCy, including `en_core_web_sm` and `de_core_news_sm`
- NumPy, Matplotlib, and tqdm

The original notebooks install PyTorch 2.3.0 and TorchText 0.18.0. Use a compatible environment when reproducing them; newer TorchText/PyTorch combinations may not support the same APIs. Install the spaCy language models as well as the Python packages.

Example setup:

```bash
pip install torch==2.3.0 torchtext==0.18.0 torchvision datasets evaluate spacy numpy matplotlib tqdm
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Learning outcomes

After working through these notebooks, you should understand how to:

- Build an encoder–decoder translation model from PyTorch building blocks.
- Explain the roles of recurrent hidden states, LSTM cell states, and context vectors.
- Compare LSTM and GRU sequence models.
- Implement teacher forcing and autoregressive token generation.
- Explain why attention helps a decoder access different parts of the source sequence.
- Visualize attention alignments and evaluate generated translations with BLEU.

## Scope and notes

These are educational implementations for studying classic recurrent neural machine translation. They are not Transformer-based translation systems, and they do not use a pretrained translation model. BLEU scores and training outcomes depend on the runtime, preprocessing, package versions, and training configuration; no benchmark result is claimed here.

The notebooks use a conventional word-level vocabulary and greedy decoding. They are intended as a foundation for later study of subword tokenization, beam search, Transformer architectures, and modern large-scale translation systems.
