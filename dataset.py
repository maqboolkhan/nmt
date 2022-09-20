from torch.utils.data import Dataset
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

class nmtDataset(Dataset):
    def __init__(self, ds_path, split, train_ds=None):
        self.en = open(ds_path + split + '.en', encoding='utf-8').readlines()
        self.de = open(ds_path + split + '.de', encoding='utf-8').readlines()
        
        self.tokenizers = { 'en': get_tokenizer('spacy', language='en_core_web_sm'), 
                            'de': get_tokenizer('spacy', language='de_core_news_sm') }
        
        if split == 'train':
            self.src_vocab, self.trg_vocab = self._build_vocab()
        else:
            self.src_vocab, self.trg_vocab = train_ds.src_vocab, train_ds.trg_vocab
    
    def __len__(self):
        return len(self.en)

    def __getitem__(self, item):
        src_tokens = self.tokenizers['en'](self.en[item].lower().strip())
        trg_tokens = self.tokenizers['de'](self.de[item].lower().strip())
  
        return {
            "src": [self.src_vocab['<sos>']] + self.src_vocab.lookup_indices(src_tokens) + [self.src_vocab['<eos>']],
            "trg": [self.trg_vocab['<sos>']] + self.trg_vocab.lookup_indices(trg_tokens) + [self.trg_vocab['<eos>']]
        }
        
    def _build_vocab(self):
        src_vocab = build_vocab_from_iterator(self._get_tokens(self.en, 'en'), specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=2)  # discarding words occurs < 2 times
        trg_vocab = build_vocab_from_iterator(self._get_tokens(self.de, 'de'), specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=2)
  
        trg_vocab.set_default_index(trg_vocab['<unk>'])
        src_vocab.set_default_index(src_vocab['<unk>'])

        return src_vocab, trg_vocab
    
    def _get_tokens(self, corpus, lang):
        for line in corpus:
            yield self.tokenizers[lang](line.lower().strip())
