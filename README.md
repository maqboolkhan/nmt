### Neural Machine Translation (NMT)
---
A repository to explore different techniques for neural machine translation or Sequence-to-sequence models using Pytorch. <br>
All models are written from scratch in order to be easily understood and integrateable in any future work!

### Stats

I only ran all models for 10 epochs. 
All LSTM models are 2 layered.

| Model | BLEU (test set) | Time (seconds/epoch) GTX 1050 | Time (seconds/epoch) RTX 3070
| ---  |--- | --- | --- |
| Simple LSTM model | 0.152 | 100 | 28 |
| Bi-LSTM model | 0.208 | 141 | 36 |
| Attention with Bi-LSTM model | 0.267 | 269 | 63 |
| Transformer | 0.363 | 75 | 16 |
| Transformer with GloVe | 0.370 | 49 | 12 |

### Notes:

1. All notebooks are heavily commented. 
2. `Torchtext` comes with Multi30K dataset. For some reasons (I dont remember), I did not use it.
3. There is a notebook `lstm-exploration.ipynb` which explores how LSTM cell works in Pytorch.
4. Feel free to raise an issue or send a pull request ⚡️.