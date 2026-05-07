"""
Bahdanau Attention (RNNsearch) 구현
논문: "Neural Machine Translation by Jointly Learning to Align and Translate"
      Bahdanau, Cho, Bengio — ICLR 2015  (arXiv:1409.0473)

핵심 수식 (논문 Appendix A 기준):
  Encoder  : BiGRU  →  hj = [→hj ; ←hj]
  Attention : eij = va^T * tanh(Wa*si-1 + Ua*hj)
              αij = softmax(eij)
              ci  = Σj αij * hj
  Decoder  : GRU with context  si = (1-zi)⊙si-1 + zi⊙s̃i
  Output   : maxout → softmax
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# 1. GRU Cell — 논문 Appendix A.1.1 수식 그대로
# ──────────────────────────────────────────────
class BahdanauGRUCell(nn.Module):
    """
    논문 수식:
        z  = σ(Wz*e(y) + Uz*s + Cz*c)
        r  = σ(Wr*e(y) + Ur*s + Cr*c)
        s̃  = tanh(W*e(y) + U*(r⊙s) + C*c)
        s' = (1-z)⊙s + z⊙s̃
    context c를 받는 decoder용 GRU.
    """
    def __init__(self, input_size: int, hidden_size: int, context_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # update gate
        self.Wz = nn.Linear(input_size, hidden_size, bias=False)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Cz = nn.Linear(context_size, hidden_size, bias=True)

        # reset gate
        self.Wr = nn.Linear(input_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Cr = nn.Linear(context_size, hidden_size, bias=True)

        # candidate hidden
        self.W  = nn.Linear(input_size, hidden_size, bias=False)
        self.U  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.C  = nn.Linear(context_size, hidden_size, bias=True)

    def forward(self, embed: Tensor, s_prev: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            embed   : (B, embed_dim)  — e(yi-1)
            s_prev  : (B, H)          — si-1
            context : (B, 2H)         — ci
        Returns:
            s_next  : (B, H)
        """
        z  = torch.sigmoid(self.Wz(embed) + self.Uz(s_prev) + self.Cz(context))
        r  = torch.sigmoid(self.Wr(embed) + self.Ur(s_prev) + self.Cr(context))
        s_tilde = torch.tanh(self.W(embed) + self.U(r * s_prev) + self.C(context))
        s_next  = (1 - z) * s_prev + z * s_tilde
        return s_next


# ──────────────────────────────────────────────
# 2. Encoder — Bidirectional GRU  (Section 3.2)
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    """
    BiGRU encoder.
    hj = [→hj ; ←hj]  ∈ R^{2n}     (논문 식 7)
    초기 decoder hidden state:
        s0 = tanh(Ws * ←h1)
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.birnn = nn.GRU(
            embed_dim, hidden_size,
            batch_first=True, bidirectional=True
        )
        # s0 초기화용 (논문 Appendix A.2.2)
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src     : (B, Tx)  token ids
            src_len : (B,)     실제 길이 (padding 제외)
        Returns:
            annotations : (B, Tx, 2n)  — {h1, ..., hTx}
            s0          : (B, n)       — decoder 초기 hidden state
        """
        embed = self.dropout(self.embedding(src))           # (B, Tx, m)

        packed = nn.utils.rnn.pack_padded_sequence(
            embed, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        output, h_n = self.birnn(packed)                    # h_n: (2, B, n)
        annotations, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True
        )                                                   # (B, Tx, 2n)

        # 역방향 마지막 hidden → s0  (논문: s0 = tanh(Ws * ←h1))
        h_backward_last = h_n[1]                            # (B, n)
        s0 = torch.tanh(self.Ws(h_backward_last))           # (B, n)

        return annotations, s0


# ──────────────────────────────────────────────
# 3. Attention (Alignment Model) — Section 3.1, Appendix A.1.2
# ──────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    """
    논문 수식:
        eij   = va^T * tanh(Wa*si-1 + Ua*hj)
        αij   = softmax(eij)
        ci    = Σj αij * hj

    최적화: Ua*hj는 매 step 반복 계산 불필요 → forward에서 미리 전달받음.
    """
    def __init__(self, hidden_size: int, annot_size: int, attn_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, attn_size, bias=False)  # decoder hidden
        self.Ua = nn.Linear(annot_size,  attn_size, bias=False)  # encoder annot
        self.va = nn.Linear(attn_size, 1, bias=False)

    def forward(
        self,
        s_prev:    Tensor,           # (B, H)
        annot:     Tensor,           # (B, Tx, 2H)
        Ua_annot:  Tensor,           # (B, Tx, A)  — 미리 계산된 Ua*hj
        src_mask:  Optional[Tensor], # (B, Tx) bool, True=패딩
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            context : (B, 2H)
            alpha   : (B, Tx)    — attention weights αij
        """
        # eij = va^T * tanh(Wa*si-1 + Ua*hj)
        Wa_s = self.Wa(s_prev).unsqueeze(1)       # (B, 1, A)
        energy = self.va(torch.tanh(Wa_s + Ua_annot)).squeeze(-1)  # (B, Tx)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float('-inf'))

        alpha   = F.softmax(energy, dim=-1)        # (B, Tx)
        context = torch.bmm(alpha.unsqueeze(1), annot).squeeze(1)  # (B, 2H)
        return context, alpha


# ──────────────────────────────────────────────
# 4. Decoder — Appendix A.2.2
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    """
    매 step:
        ci  ← Attention(si-1, {hj})
        si  ← BahdanauGRUCell(e(yi-1), si-1, ci)
        ti  ← maxout(Uo*si-1 + Vo*e(yi-1) + Co*ci)   (deep output)
        p(yi) ∝ exp(yi^T * Wo * ti)
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_size: int,
        annot_size:  int,  # = 2 * enc_hidden
        attn_size:   int,
        maxout_size: int,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention  = BahdanauAttention(hidden_size, annot_size, attn_size)
        self.gru_cell   = BahdanauGRUCell(embed_dim, hidden_size, annot_size)
        self.dropout    = nn.Dropout(dropout)

        # deep output with maxout (논문 Appendix A.2.2)
        # t̃i = Uo*si-1 + Vo*e(yi-1) + Co*ci   shape: (B, 2*maxout_size)
        # ti  = max_{pairs}(t̃i)                 shape: (B, maxout_size)
        self.Uo = nn.Linear(hidden_size, 2 * maxout_size, bias=False)
        self.Vo = nn.Linear(embed_dim,   2 * maxout_size, bias=False)
        self.Co = nn.Linear(annot_size,  2 * maxout_size, bias=True)
        self.Wo = nn.Linear(maxout_size, vocab_size, bias=False)

    def _precompute_Ua(self, annot: Tensor) -> Tensor:
        """Ua*hj 를 디코딩 전에 한번만 계산 (효율화)."""
        return self.attention.Ua(annot)   # (B, Tx, A)

    def forward_step(
        self,
        token:     Tensor,           # (B,)  이전 token
        s_prev:    Tensor,           # (B, H)
        annot:     Tensor,           # (B, Tx, 2H)
        Ua_annot:  Tensor,           # (B, Tx, A)
        src_mask:  Optional[Tensor], # (B, Tx)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        embed   = self.dropout(self.embedding(token))     # (B, m)
        context, alpha = self.attention(s_prev, annot, Ua_annot, src_mask)
        s_next  = self.gru_cell(embed, s_prev, context)  # (B, H)

        # deep output + maxout
        t_tilde = self.Uo(s_prev) + self.Vo(embed) + self.Co(context)  # (B, 2L)
        B, two_L = t_tilde.shape
        t_i = t_tilde.view(B, two_L // 2, 2).max(dim=-1).values        # (B, L) maxout

        logit = self.Wo(t_i)                              # (B, vocab_size)
        return logit, s_next, alpha

    def forward(
        self,
        tgt:       Tensor,           # (B, Ty)  teacher forcing input
        s0:        Tensor,           # (B, H)
        annot:     Tensor,           # (B, Tx, 2H)
        src_mask:  Optional[Tensor], # (B, Tx)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            logits : (B, Ty, vocab_size)
            alphas : (B, Ty, Tx)
        """
        Ua_annot = self._precompute_Ua(annot)
        s = s0
        logits, alphas = [], []

        for t in range(tgt.size(1)):
            logit, s, alpha = self.forward_step(
                tgt[:, t], s, annot, Ua_annot, src_mask
            )
            logits.append(logit)
            alphas.append(alpha)

        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


# ──────────────────────────────────────────────
# 5. Full Seq2Seq (RNNsearch)
# ──────────────────────────────────────────────
class RNNsearch(nn.Module):
    """
    논문의 RNNsearch 모델 전체.
    src_vocab, tgt_vocab : 어휘 크기
    embed_dim            : word embedding 차원  (논문: 620)
    hidden_size          : GRU hidden 차원      (논문: 1000)
    attn_size            : alignment model 차원 (논문: 1000)
    maxout_size          : maxout layer 크기    (논문: 500)
    """
    def __init__(
        self,
        src_vocab:   int,
        tgt_vocab:   int,
        embed_dim:   int = 256,
        hidden_size: int = 512,
        attn_size:   int = 512,
        maxout_size: int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_size, dropout)
        self.decoder = Decoder(
            tgt_vocab, embed_dim, hidden_size,
            annot_size=2 * hidden_size,
            attn_size=attn_size,
            maxout_size=maxout_size,
            dropout=dropout,
        )

    def forward(
        self,
        src:     Tensor,            # (B, Tx)
        src_len: Tensor,            # (B,)
        tgt:     Tensor,            # (B, Ty)  — teacher forcing (SOS 포함)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            logits : (B, Ty, tgt_vocab)
            alphas : (B, Ty, Tx)
        """
        src_mask = (src == 0)       # padding mask  (True = 패딩)
        annotations, s0 = self.encoder(src, src_len)
        logits, alphas  = self.decoder(tgt, s0, annotations, src_mask)
        return logits, alphas

    @torch.no_grad()
    def translate(
        self,
        src:     Tensor,            # (1, Tx)
        src_len: Tensor,            # (1,)
        sos_id:  int,
        eos_id:  int,
        max_len: int = 50,
    ) -> Tuple[list, Tensor]:
        """Greedy decoding (beam search는 별도 구현 필요)."""
        self.eval()
        src_mask = (src == 0)
        annotations, s = self.encoder(src, src_len)
        Ua_annot = self.decoder._precompute_Ua(annotations)

        tokens, alphas = [sos_id], []
        token = torch.tensor([sos_id], device=src.device)

        for _ in range(max_len):
            logit, s, alpha = self.decoder.forward_step(
                token, s, annotations, Ua_annot, src_mask
            )
            pred = logit.argmax(-1).item()
            tokens.append(pred)
            alphas.append(alpha.squeeze(0).cpu())
            if pred == eos_id:
                break
            token = torch.tensor([pred], device=src.device)

        return tokens, torch.stack(alphas)


# ──────────────────────────────────────────────
# 6. 간단한 동작 확인 (toy example)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 (논문 설정 축소판)
    SRC_VOCAB, TGT_VOCAB = 100, 80
    EMBED     = 64
    HIDDEN    = 128
    ATTN      = 128
    MAXOUT    = 64
    BATCH     = 4
    SRC_LEN   = 12
    TGT_LEN   = 10

    model = RNNsearch(
        src_vocab=SRC_VOCAB, tgt_vocab=TGT_VOCAB,
        embed_dim=EMBED, hidden_size=HIDDEN,
        attn_size=ATTN, maxout_size=MAXOUT,
        dropout=0.1,
    ).to(device)

    # 파라미터 수 출력
    total = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total:,}")

    # ── Forward pass ──
    src     = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    src_len = torch.tensor([SRC_LEN] * BATCH)
    tgt_in  = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)  # teacher forcing
    tgt_out = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)  # 정답

    logits, alphas = model(src, src_len, tgt_in)
    print(f"logits shape : {logits.shape}")   # (B, Ty, tgt_vocab)
    print(f"alphas shape : {alphas.shape}")   # (B, Ty, Tx)

    # ── Loss 계산 (Cross-Entropy) ──
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(logits.reshape(-1, TGT_VOCAB), tgt_out.reshape(-1))
    print(f"초기 loss    : {loss.item():.4f}")

    # ── Backprop 확인 ──
    loss.backward()
    print("Backprop     : OK")

    # ── 간단한 학습 루프 (10 step) ──
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)
    model.train()
    print("\n── 10 step 학습 ──")
    for step in range(10):
        optimizer.zero_grad()
        logits, _ = model(src, src_len, tgt_in)
        loss = loss_fn(logits.reshape(-1, TGT_VOCAB), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 논문 gradient clipping
        optimizer.step()
        if (step + 1) % 2 == 0:
            print(f"  step {step+1:2d} | loss: {loss.item():.4f}")

    # ── Greedy decode (inference) ──
    model.eval()
    src_single  = src[:1]
    len_single  = src_len[:1]
    tokens, att = model.translate(src_single, len_single, sos_id=1, eos_id=2, max_len=15)
    print(f"\nGreedy decode: {tokens}")
    print(f"Attention map: {att.shape}")  # (Ty, Tx)
    print("\n구현 완료!")
