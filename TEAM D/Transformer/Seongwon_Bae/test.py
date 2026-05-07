import torch
from model import TransformerModel
from data import build_data

def translate_sentence(model, sentence, vocabs, device, max_length=50):
    model.eval()
    tokens = [vocabs['src']['<bos>']] + vocabs['src'](sentence) + [vocabs['src']['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    outputs = [vocabs['trg']['<bos>']]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        
        if best_guess == vocabs['trg']['<eos>']:
            break
            
    return [vocabs['trg'].get_itos()[idx] for idx in outputs]
