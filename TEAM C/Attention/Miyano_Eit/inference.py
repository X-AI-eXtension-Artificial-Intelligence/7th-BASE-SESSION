import torch
from model import Seq2SeqAttention
from data_loader import Vocabulary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
os.makedirs("etc", exist_ok=True)

def translate(model, sentence, src_vocab, tgt_vocab, max_len=50, device="cpu"):
    model.eval()

    src = src_vocab.encode(sentence, max_len)
    src_tensor = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    tgt_idx = tgt_vocab.word2idx["<SOS>"]
    translated = []
    attentions = []

    for _ in range(max_len):
        y_prev = torch.tensor([tgt_idx], dtype=torch.long).to(device)

        with torch.no_grad():
            pred, hidden, alpha = model.decoder(y_prev, hidden, encoder_outputs)

        attentions.append(alpha.squeeze(0).cpu())
        tgt_idx = pred.argmax(1).item()

        if tgt_idx == tgt_vocab.word2idx["<EOS>"]:
            break

        translated.append(tgt_vocab.idx2word.get(tgt_idx, "<UNK>"))

    return translated, attentions


def plot_attention(src_sentence, translation, attentions):
    src_tokens = src_sentence.split()
    attn_matrix = torch.stack(attentions).numpy()   # (tgt_len, src_len)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap="Blues")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(translation)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(translation)

    ax.set_xlabel("Source")
    ax.set_ylabel("Translation")
    ax.set_title("Attention Weights")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("etc/attention_map.png", dpi=150)
    plt.show()
    print("Attention map 저장 완료 → etc/attention_map.png")


if __name__ == "__main__":
    # vocab은 실제로는 저장된 것을 로드해야 함
    # 여기서는 간단히 main.py와 동일하게 재구성
    from data_loader import Vocabulary

    src_sentences = [
        "the cat sat on the mat",
        "a dog runs in the park",
        "she reads a book every day",
    ]
    tgt_sentences = [
        "le chat s assis sur le tapis",
        "un chien court dans le parc",
        "elle lit un livre chaque jour",
    ]

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(src_sentences)
    tgt_vocab.build(tgt_sentences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqAttention(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=64,
        hidden_dim=128,
    ).to(device)

    model.load_state_dict(torch.load("etc/model.pt", map_location=device))

    sentence = "the cat sat on the mat"
    translation, attentions = translate(model, sentence, src_vocab, tgt_vocab, device=device)

    print(f"Input:       {sentence}")
    print(f"Translation: {' '.join(translation)}")

    plot_attention(sentence, translation, attentions)