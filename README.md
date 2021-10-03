### Neural Machine Translation (NMT)
---
A repository to explore different techniques for machine translation.

Progess so far! (Only with 10 epochs, All LSTM models are 2 layered and I used GTX 1050 ti)


| Model | BLEU (test set) | Time (seconds/epoch) |
| ---  |--- | --- |
| Simple LSTM model | 0.150 | 142 |
| Bi-LSTM model | 0.195 | 141 |
| Attention with Bi-LSTM model | 0.232 | 309 |