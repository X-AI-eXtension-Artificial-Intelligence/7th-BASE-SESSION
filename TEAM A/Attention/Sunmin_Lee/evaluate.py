import torch
from config import Config
from data.dataset import get_dataloader
from model import Encoder, BahdanauAttention, Decoder
from utils.visualize import plot_attention, print_prediction


def evaluate():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    encoder = Encoder(cfg.input_size, cfg.embed_dim, cfg.hidden_size)
    decoder = Decoder(cfg.output_size, cfg.embed_dim, cfg.hidden_size)

    encoder.load_state_dict(torch.load("encoder.pt", map_location="cpu"))
    decoder.load_state_dict(torch.load("decoder.pt", map_location="cpu"))
    encoder.eval()
    decoder.eval()

    loader = get_dataloader(100, cfg.seq_len, cfg.input_size,
                            cfg.batch_size, shuffle=False)
    src_batch, tgt_batch = next(iter(loader))

    with torch.no_grad():
        enc_outputs, dec_hidden = encoder(src_batch)
        dec_input = torch.zeros(src_batch.size(0), dtype=torch.long)

        all_preds  = []
        all_alphas = []

        for t in range(cfg.seq_len):
            pred, dec_hidden, alpha = decoder(dec_input, dec_hidden, enc_outputs)
            dec_input = pred.argmax(dim=1)
            all_preds.append(dec_input)
            all_alphas.append(alpha)

    # 샘플 3개 출력
    for idx in range(3):
        src_seq  = src_batch[idx].tolist()
        tgt_seq  = tgt_batch[idx].tolist()
        pred_seq = [all_preds[t][idx].item() for t in range(cfg.seq_len)]
        attn_mat = torch.stack(
            [all_alphas[t][idx] for t in range(cfg.seq_len)]
        )

        print_prediction(src_seq, tgt_seq, pred_seq)
        plot_attention(src_seq, pred_seq, attn_mat,
                       save_path=f"attention_map_{idx}.png")


if __name__ == "__main__":
    evaluate()