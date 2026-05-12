import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from data import build_data
from util import save_checkpoint

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    
    train_loader, vocabs = build_data(batch_size)
    
    src_vocab_size = len(vocabs['src'])
    trg_vocab_size = len(vocabs['trg'])
    
    model = TransformerModel(
        src_vocab_size, trg_vocab_size, 
        vocabs['src']['<pad>'], vocabs['trg']['<pad>']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabs['trg']['<pad>'])

    model.train()
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            
            # target input과 output 분리
            output = model(src, trg[:, :-1])
            output = output.reshape(-1, output.shape[2])
            trg_target = trg[:, 1:].reshape(-1)
            
            optimizer.zero_grad()
            loss = criterion(output, trg_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {sum(losses)/len(losses):.4f}")
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()})

if __name__ == "__main__":
    train()