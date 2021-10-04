import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

class Logger:
    def __init__(self, path):
        self.__f = open(path, 'w')
    
    def log(self, msg):
        print(msg)
        self.__f.write(msg+'\n')
    
    def close(self):
        self.__f.close()

def collate_fn(batch, device):
    PAD_IDX = 1
    trgs = []
    srcs = []
    for row in batch:
        srcs.append(torch.tensor(row["src"]).to(device))
        trgs.append(torch.tensor(row["trg"]).to(device))

    padded_srcs = pad_sequence(srcs, padding_value=PAD_IDX)
    padded_trgs = pad_sequence(trgs, padding_value=PAD_IDX)
    return {"src": padded_srcs, "trg": padded_trgs}


def translate(snt, dataset, model, attention, device):
    tokens = dataset.tokenizers['en'](snt.lower().strip())
    indices = [dataset.src_vocab['<sos>']] + dataset.src_vocab.lookup_indices(tokens) + [dataset.src_vocab['<eos>']]
    inp_tensor = torch.tensor(indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        if attention:
            eouts, hidden, cell = model.Encoder(inp_tensor)
        else:
            hidden, cell = model.Encoder(inp_tensor)

    outputs = [dataset.trg_vocab["<sos>"]]

    for _ in range(50):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            if attention:
                output, hidden, cell = model.Decoder(previous_word, eouts, hidden, cell)
            else:
                output, hidden, cell = model.Decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # M odel predicts it's the end of the sentence
        if output.argmax(1).item() == dataset.trg_vocab['<eos>']:
            break

    return dataset.trg_vocab.lookup_tokens(outputs)


def bleu(model, dataset, attention, device):
    targets = []
    outputs = []

    for example in tqdm(dataset):
        src = example["src"][1:-1]
        trg = example["trg"][1:-1]
        
        src = ' '.join(dataset.src_vocab.lookup_tokens(src))
        trg = dataset.trg_vocab.lookup_tokens(trg)

        prediction = translate(src, dataset, model, attention, device)
        prediction = prediction[1:-1]  # remove <eos> token
        
        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def evaluate_model(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            src = batch["src"]  # shape --> e.g. (19, 2) sentence len, batch size
            trg = batch["trg"]  # shape --> e.g. (3, 2) sentence len, batch size

            # Pass the input and target for model's forward method
            # Shape --> (sentence len of TRG, batch size, vocab size) e.g (3, 2, 196)
            # Explanation:
            #    It just outputs probabilities for every single word in our vocab
            #    for each word in sentence and each sentence in batch size
            output = model(src, trg, 0)

            # Updating output shape --> [sentence length * batch size , vocab size]
            # e.g (6, 196)
            output = output.reshape(-1, output.shape[2])

            # sentence len  * batch size
            target = trg.reshape(-1)

            # Calculate the loss value for every epoch
            loss = criterion(output, target)

            epoch_loss += loss.detach().cpu()

    return epoch_loss/len(data_loader)

def train_model(model, data_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        src = batch["src"]  # shape --> e.g. (19, 2) sentence len, batch size
        trg = batch["trg"]  # shape --> e.g. (3, 2) sentence len, batch size

        # Clear the accumulating gradients
        optimizer.zero_grad()

        # Pass the input and target for model's forward method
        # Shape --> (sentence len of TRG, batch size, vocab size) e.g (3, 2, 196)
        # Explanation:
        #    It just outputs probabilities for every single word in our vocab
        #    for each word in sentence and each sentence in batch size
        output = model(src, trg)

        # Updating output shape --> [sentence length * batch size , vocab size]
        # e.g (6, 196)
        output = output.reshape(-1, output.shape[2])

        # sentence len  * batch size
        target = trg.reshape(-1)

        # Calculate the loss value for every epoch
        loss = criterion(output, target)

        # Calculate the gradients for weights & biases using back-propagation
        loss.backward()

        # Clip the gradient value is it exceeds > 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Update the weights values
        optimizer.step()
        
        epoch_loss += loss.detach().cpu()
    
    return epoch_loss / len(data_loader)