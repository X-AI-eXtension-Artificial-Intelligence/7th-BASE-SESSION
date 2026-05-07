import torch
import matplotlib.pyplot as plt
from model import Transformer
from data_loader import Vocabulary


def translate(model, sentence, src_vocab, tgt_vocab, max_len=50, device="cpu"):
    model.eval()

    src = src_vocab.encode(sentence, max_len)
    src_tensor = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)

    tgt_idx = [tgt_vocab.word2idx["<SOS>"]]
    translated = []
    all_attn_weights = []

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_idx, dtype=torch.long).unsqueeze(0).to(device)

            output, attn_weights = model(src_tensor, tgt_tensor)

            next_token = output[0, -1, :].argmax().item()
            tgt_idx.append(next_token)

            last_layer_attn = attn_weights[-1][0]
            all_attn_weights.append(last_layer_attn[:, -1, :].mean(0).cpu())

            if next_token == tgt_vocab.word2idx["<EOS>"]:
                break

            translated.append(tgt_vocab.idx2word.get(next_token, "<UNK>"))

    return translated, all_attn_weights


def plot_attention(src_sentence, translation, attn_weights):
    src_tokens = src_sentence.split()
    attn_matrix = torch.stack(attn_weights).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap="Blues")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(translation)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(translation)

    ax.set_xlabel("Source")
    ax.set_ylabel("Translation")
    ax.set_title("Transformer Cross-Attention Weights (last layer, avg heads)")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("etc/transformer_attention_map.png", dpi=150)
    plt.show()
    print("Attention map 저장 완료 → etc/transformer_attention_map.png")


if __name__ == "__main__":
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

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        h=4,
        d_ff=256,
        N=3,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load("etc/transformer.pt", map_location=device))

    sentence = "i want to eat ramen"
    translation, attn_weights = translate(
        model, sentence, src_vocab, tgt_vocab, device=device
    )

    print(f"Input:       {sentence}")
    print(f"Translation: {' '.join(translation)}")

    plot_attention(sentence, translation, attn_weights)