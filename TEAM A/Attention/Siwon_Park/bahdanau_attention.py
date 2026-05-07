"""
Bahdanau Attention 구현 (Bahdanau et al., ICLR 2015)
"Neural Machine Translation by Jointly Learning to Align and Translate"

이 코드는 논문을 공부하면서 이해한 내용을 바탕으로 작성되었습니다.
논문의 핵심 아이디어:
  - 기존 Encoder-Decoder는 문장 전체를 하나의 고정 길이 벡터에 압축
    → 긴 문장에서 정보 손실 (Bottleneck Problem)
  - 해결책: 매 번역 스텝마다 원문에서 관련 있는 부분을 동적으로 골라서 참고
    → 이게 바로 Attention Mechanism

실행 환경: Python 3.8+, PyTorch
설치: pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ──────────────────────────────────────────────────────────────
# 1. 인코더: Bidirectional RNN (BiRNN)
# ──────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    논문 Section 3.2 + Appendix A.2.1 구현

    왜 양방향(Bidirectional)인가?
    - 단방향 RNN은 앞→뒤로만 읽음
      → '나는 은행에 갔다'에서 '은행'을 처리할 때 뒤에 '갔다'를 아직 모름
    - BiRNN은 앞→뒤(순방향) + 뒤→앞(역방향) 두 방향으로 동시에 읽음
      → 각 단어의 어노테이션 hⱼ가 앞뒤 문맥을 모두 담게 됨

    출력:
    - outputs: 각 단어별 어노테이션 hⱼ = [h→ⱼ ; h←ⱼ]  shape: (batch, seq_len, hidden*2)
    - hidden:  마지막 은닉 상태 (디코더 초기값 s₀ 계산에 사용)
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, dropout=0.1):
        super().__init__()

        # 단어 임베딩: 1-of-K 벡터 → 밀집 벡터(dense vector)로 변환
        # 논문에서 임베딩 차원 m = 620
        # 순방향/역방향 RNN이 이 임베딩 행렬 E를 공유함 (Appendix A.2.1)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 양방향 GRU
        # GRU = 논문의 Gated Hidden Unit과 거의 동일한 구조
        # (논문은 자체 구현한 gated unit을 쓰지만 GRU로 근사 가능)
        #
        # Appendix A.1.1에서 설명한 두 가지 게이트:
        #   - 업데이트 게이트 zᵢ: "이전 기억을 얼마나 유지할까?"
        #   - 리셋 게이트 rᵢ:   "이전 기억을 얼마나 잊을까?"
        # sᵢ = (1 − zᵢ) ∘ sᵢ₋₁ + zᵢ ∘ s̃ᵢ
        # → 이전 상태와 새 정보를 zᵢ 비율로 믹스
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            bidirectional=True,   # 순방향 + 역방향 동시 처리
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        """
        Args:
            src:         원문 토큰 인덱스  shape: (batch, src_len)
            src_lengths: 각 문장의 실제 길이 (패딩 제외)

        Returns:
            outputs: 모든 단어의 어노테이션 hⱼ  shape: (batch, src_len, hidden*2)
            hidden:  마지막 은닉 상태           shape: (batch, hidden*2)
        """
        # Step 1: 단어 인덱스 → 임베딩 벡터
        # xⱼ ∈ ℝᵐ (논문 Appendix A.2.1)
        embedded = self.dropout(self.embedding(src))
        # shape: (batch, src_len, embed_dim)

        # Step 2: 패딩 부분을 RNN 연산에서 제외 (효율적 처리)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Step 3: 양방향 GRU 통과
        # 순방향: h→₁, h→₂, ..., h→_T
        # 역방향: h←₁, h←₂, ..., h←_T
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs shape: (batch, src_len, hidden*2)
        # 각 outputs[b, j, :] = [h→ⱼ ; h←ⱼ] — 순방향과 역방향을 연결(concatenate)한 것

        # Step 4: 디코더 초기 상태 s₀ 준비
        # 논문 Appendix A.2.2:
        #   s₀ = tanh(Ws · h←₁)
        #   h←₁ = 역방향 RNN이 문장 전체를 읽고 난 최종 상태
        #        = 문장 전체의 요약 정보를 담고 있음
        # hidden shape: (num_directions*num_layers, batch, hidden)
        #   hidden[0] = 순방향 마지막 상태
        #   hidden[1] = 역방향 마지막 상태
        fwd_hidden = hidden[0]  # 순방향 최종 은닉 상태
        bwd_hidden = hidden[1]  # 역방향 최종 은닉 상태 ← 이게 h←₁

        # 순방향과 역방향을 이어붙여서 디코더에 전달
        hidden = torch.cat([fwd_hidden, bwd_hidden], dim=1)
        # shape: (batch, hidden*2)

        return outputs, hidden


# ──────────────────────────────────────────────────────────────
# 2. 정렬 모델 (Alignment Model) = Attention
# ──────────────────────────────────────────────────────────────
class Attention(nn.Module):
    """
    논문 Section 3.1 + Appendix A.1.2 구현

    핵심 역할:
    "지금 번역할 단어를 생성하려면 원문의 어떤 단어를 봐야 할까?"를 계산

    수식:
        eᵢⱼ  = vₐᵀ · tanh(Wₐ · sᵢ₋₁ + Uₐ · hⱼ)   ← 정렬 에너지 (관련도 점수)
        αᵢⱼ  = softmax(eᵢⱼ)                          ← 어텐션 가중치 (0~1, 합=1)
        cᵢ   = Σ αᵢⱼ · hⱼ                            ← 컨텍스트 벡터 (가중합)

    중요한 최적화 (Appendix A.1.2):
        Uₐ · hⱼ 는 디코더 스텝 i와 무관하게 항상 같음
        → 미리 계산해두면 매 스텝마다 반복 계산 불필요
        → 이 구현에서는 encoder_out을 미리 선형 변환해서 전달

    αᵢⱼ의 의미 (논문 Section 3.1):
        αᵢⱼ ≈ "목표 단어 yᵢ가 원문 단어 xⱼ로부터 번역될 확률"
        → Figure 3의 정렬 행렬 시각화가 바로 이 값들을 그린 것
        → 대각선 패턴: 영어-프랑스어 어순이 비슷한 부분
        → 비대각선 패턴: 형용사-명사 순서 역전 등 비단조적 정렬
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Wₐ: 디코더 은닉 상태 sᵢ₋₁을 변환하는 행렬
        # 입력 차원: hidden*2 (인코더가 양방향이므로), 출력: hidden
        self.Wa = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Uₐ: 인코더 어노테이션 hⱼ를 변환하는 행렬
        # 입력 차원: hidden*2 (BiRNN 출력), 출력: hidden
        # 이 변환은 디코더 스텝과 무관 → forward() 밖에서 미리 계산 가능
        self.Ua = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # vₐ: 최종 에너지 스칼라값을 만드는 벡터
        # tanh 결과(hidden 차원)를 스칼라 하나로 줄임
        self.va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, encoder_outputs_transformed, mask=None):
        """
        Args:
            decoder_hidden:              현재 디코더 은닉 상태 sᵢ₋₁
                                         shape: (batch, hidden*2)
            encoder_outputs:             인코더 어노테이션 hⱼ (원본)
                                         shape: (batch, src_len, hidden*2)
            encoder_outputs_transformed: Uₐ·hⱼ를 미리 계산해둔 값 (최적화)
                                         shape: (batch, src_len, hidden)
            mask:                        패딩 위치를 -inf로 마스킹
                                         shape: (batch, src_len)

        Returns:
            context:  컨텍스트 벡터 cᵢ   shape: (batch, hidden*2)
            alpha:    어텐션 가중치 αᵢⱼ  shape: (batch, src_len)
        """
        # Step 1: Wₐ · sᵢ₋₁  계산
        # decoder_hidden: (batch, hidden*2) → (batch, hidden)
        decoder_transformed = self.Wa(decoder_hidden)
        # shape: (batch, hidden)

        # 브로드캐스팅을 위해 차원 확장
        # (batch, hidden) → (batch, 1, hidden)
        decoder_transformed = decoder_transformed.unsqueeze(1)

        # Step 2: eᵢⱼ = vₐᵀ · tanh(Wₐ·sᵢ₋₁ + Uₐ·hⱼ)
        # encoder_outputs_transformed: (batch, src_len, hidden)
        # decoder_transformed:         (batch, 1,       hidden)  → 자동 브로드캐스팅
        energy = self.va(torch.tanh(decoder_transformed + encoder_outputs_transformed))
        # shape: (batch, src_len, 1) → squeeze → (batch, src_len)
        energy = energy.squeeze(2)

        # Step 3: 패딩 위치는 어텐션을 주면 안 됨 → -∞ 처리
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Step 4: αᵢⱼ = softmax(eᵢⱼ)
        # 모든 원문 단어에 대한 가중치의 합 = 1
        # 논문: "αᵢⱼ는 확률로 해석 가능"
        alpha = F.softmax(energy, dim=1)
        # shape: (batch, src_len)

        # Step 5: cᵢ = Σ αᵢⱼ · hⱼ  (어노테이션들의 가중합)
        # alpha:           (batch, src_len)    → unsqueeze → (batch, 1, src_len)
        # encoder_outputs: (batch, src_len, hidden*2)
        # bmm: 배치 행렬 곱 → (batch, 1, hidden*2) → squeeze → (batch, hidden*2)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        # shape: (batch, hidden*2)

        return context, alpha


# ──────────────────────────────────────────────────────────────
# 3. 디코더
# ──────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    논문 Section 3.1 + Appendix A.2.2 구현

    매 스텝마다:
    1. 직전에 생성한 단어 yᵢ₋₁을 임베딩
    2. Attention으로 현재 스텝의 컨텍스트 벡터 cᵢ 계산
    3. [yᵢ₋₁ 임베딩 ; cᵢ]를 GRU에 입력 → 새 은닉 상태 sᵢ
    4. sᵢ, yᵢ₋₁, cᵢ를 모두 사용해서 다음 단어 yᵢ 예측

    기존 Encoder-Decoder와의 결정적 차이:
    - 기존: 모든 스텝에서 동일한 고정 벡터 c를 사용
    - 제안: 매 스텝마다 cᵢ가 새로 계산됨 (동적 컨텍스트)
      → 긴 문장에서도 원문의 관련 부분을 항상 다시 참고 가능
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)

        # 목표 언어 단어 임베딩
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 디코더 GRU
        # 입력: [임베딩(embed_dim) ; 컨텍스트 벡터(hidden*2)]를 이어붙인 것
        # → 매 스텝마다 컨텍스트가 달라지므로 GRU 입력도 달라짐
        self.rnn = nn.GRU(
            input_size=embed_dim + hidden_size * 2,
            hidden_size=hidden_size * 2,
            batch_first=True,
        )

        # 초기 상태 변환: s₀ = tanh(Ws · h←₁)
        # 인코더 최종 상태(hidden*2) → 디코더 초기 상태(hidden*2)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size * 2)

        # 출력층: sᵢ + yᵢ₋₁ + cᵢ를 모두 합쳐서 다음 단어 예측
        # 논문 Appendix A.2.2의 deep output 구조를 단순화한 버전
        # 원래 논문은 maxout을 쓰지만 여기서는 Linear + ReLU로 근사
        self.fc_out = nn.Linear(hidden_size * 2 * 3 + embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs,
                encoder_outputs_transformed, mask):
        """
        한 스텝(단어 하나)의 디코딩

        Args:
            tgt_token:                   직전 생성 단어 yᵢ₋₁  shape: (batch,)
            decoder_hidden:              이전 은닉 상태 sᵢ₋₁  shape: (batch, hidden*2)
            encoder_outputs:             인코더 어노테이션     shape: (batch, src_len, hidden*2)
            encoder_outputs_transformed: Uₐ·hⱼ 미리 계산      shape: (batch, src_len, hidden)
            mask:                        패딩 마스크

        Returns:
            pred:           단어 예측 로짓     shape: (batch, vocab_size)
            decoder_hidden: 새 은닉 상태 sᵢ   shape: (batch, hidden*2)
            alpha:          어텐션 가중치 αᵢⱼ shape: (batch, src_len)
        """
        # Step 1: 직전 단어 임베딩 ----- yᵢ₋₁
        # e(yᵢ₋₁) ∈ ℝᵐ (논문 수식)
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        # shape: (batch, 1, embed_dim)

        # Step 2: Attention으로 현재 스텝의 컨텍스트 벡터 cᵢ 계산
        # αᵢⱼ = softmax(vₐᵀ · tanh(Wₐ·sᵢ₋₁ + Uₐ·hⱼ))
        # cᵢ  = Σ αᵢⱼ · hⱼ
        context, alpha = self.attention(
            decoder_hidden, encoder_outputs, encoder_outputs_transformed, mask
        )
        # context shape: (batch, hidden*2)

        # Step 3: [임베딩 ; 컨텍스트]를 이어붙여서 GRU 입력 구성 ---- sᵢ
        # 논문: sᵢ = f(sᵢ₋₁, yᵢ₋₁, cᵢ)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # shape: (batch, 1, embed_dim + hidden*2)

        # Step 4: GRU로 새 은닉 상태 sᵢ 계산
        output, hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        # output shape:  (batch, 1, hidden*2)
        # hidden shape:  (1, batch, hidden*2)

        output = output.squeeze(1)
        decoder_hidden = hidden.squeeze(0)
        # 이제 decoder_hidden = sᵢ  shape: (batch, hidden*2)

        # Step 5: 다음 단어 예측
        # 논문 Appendix A.2.2:
        #   p(yᵢ | sᵢ, yᵢ₋₁, cᵢ) 를 계산
        #   원래: deep output with maxout
        #   여기서는 단순화: [sᵢ ; yᵢ₋₁ ; cᵢ]를 이어붙여서 Linear 통과
        pred = self.fc_out(
            torch.cat([output, embedded.squeeze(1), context], dim=1)
        )
        # shape: (batch, vocab_size)

        return pred, decoder_hidden, alpha


# ──────────────────────────────────────────────────────────────
# 4. Seq2Seq 전체 모델
# ──────────────────────────────────────────────────────────────
class BahdanauSeq2Seq(nn.Module):
    """
    논문 전체 구조 통합

    전체 흐름:
    1. 인코더: 원문 → 어노테이션 시퀀스 {hⱼ}
    2. 디코더: 매 스텝마다
       a. Attention으로 컨텍스트 벡터 cᵢ 계산
       b. cᵢ와 직전 단어로 새 상태 sᵢ 생성
       c. sᵢ로 다음 단어 예측
    3. 전체 시스템이 End-to-End로 함께 학습됨
       (정렬 모델 a(·)도 포함, 논문 Section 3.1)

    Teacher Forcing (학습 시):
    - 실제 정답 단어를 다음 스텝 입력으로 사용
    - 학습 초기에 안정적으로 수렴하도록 도움
    - teacher_forcing_ratio로 비율 조절 가능
    """

    def __init__(self, encoder, decoder, src_pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src):
        """
        패딩 토큰(0) 위치를 False로 표시하는 마스크 생성
        → Attention에서 패딩 위치에 가중치를 주지 않도록
        """
        mask = (src != self.src_pad_idx)
        # shape: (batch, src_len)
        return mask

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src:                   원문 토큰       shape: (batch, src_len)
            src_lengths:           원문 실제 길이
            tgt:                   목표 번역 토큰  shape: (batch, tgt_len)
            teacher_forcing_ratio: Teacher Forcing 비율 (학습 시 0.5 권장)

        Returns:
            outputs: 각 스텝의 예측 로짓  shape: (tgt_len-1, batch, vocab_size)
            attentions: 어텐션 가중치     shape: (tgt_len-1, batch, src_len)
        """
        batch_size = src.shape[0]
        tgt_len    = tgt.shape[1]
        tgt_vocab  = self.decoder.fc_out.out_features

        # 결과 저장 텐서
        outputs    = torch.zeros(tgt_len - 1, batch_size, tgt_vocab).to(src.device)
        attentions = torch.zeros(tgt_len - 1, batch_size, src.shape[1]).to(src.device)

        # ── 인코딩 ─────────────────────────────────────
        # 원문 → 어노테이션 시퀀스 {h₁, h₂, ..., h_Tx}
        # 각 hⱼ = [h→ⱼ ; h←ⱼ] (양방향 연결)
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)

        # 패딩 마스크 생성
        mask = self.create_mask(src)

        # Attention 최적화: Uₐ·hⱼ를 미리 한 번만 계산
        # 이 값은 디코딩 스텝 i가 바뀌어도 변하지 않음
        # → 매 스텝 반복 계산 방지 (Appendix A.1.2)
        encoder_outputs_transformed = self.decoder.attention.Ua(encoder_outputs)

        # ── 디코더 초기 상태 s₀ 설정 ───────────────────
        # 논문 Appendix A.2.2:
        #   s₀ = tanh(Ws · h←₁)
        #   역방향 RNN의 최종 상태 = 문장 전체의 요약
        decoder_hidden = torch.tanh(self.decoder.fc_hidden(encoder_hidden))

        # ── 디코딩 (스텝별) ─────────────────────────────
        # 첫 입력: <SOS> 토큰
        input_token = tgt[:, 0]

        for t in range(tgt_len - 1):
            # 한 스텝 디코딩:
            #   1. Attention → cᵢ 계산
            #   2. GRU → sᵢ 계산
            #   3. 다음 단어 예측
            pred, decoder_hidden, alpha = self.decoder(
                input_token,
                decoder_hidden,
                encoder_outputs,
                encoder_outputs_transformed,
                mask
            )

            outputs[t]    = pred
            attentions[t] = alpha

            # Teacher Forcing:
            # - True(확률 teacher_forcing_ratio): 정답 단어를 다음 입력으로
            # - False: 모델이 예측한 단어를 다음 입력으로
            use_teacher = random.random() < teacher_forcing_ratio
            input_token  = tgt[:, t + 1] if use_teacher else pred.argmax(1)

        return outputs, attentions


# ──────────────────────────────────────────────────────────────
# 5. 학습 설정 (Appendix B 기반)
# ──────────────────────────────────────────────────────────────
def build_model(src_vocab_size, tgt_vocab_size,
                embed_dim=256, hidden_size=512, dropout=0.1):
    """
    논문 Appendix A.2.3 하이퍼파라미터:
        은닉층 크기 n = 1000  (여기서는 메모리 절약을 위해 512로 축소)
        임베딩 차원 m = 620   (여기서는 256으로 축소)

    실제 논문 크기로 실행하려면:
        embed_dim=620, hidden_size=1000
    """
    encoder = Encoder(src_vocab_size, embed_dim, hidden_size, dropout)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_size, dropout)
    model   = BahdanauSeq2Seq(encoder, decoder)

    # 파라미터 초기화 (Appendix B.1)
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'rnn' in name:
                # 순환 가중치 → 직교 행렬 초기화
                # "역전파 시 기울기 크기가 변하지 않도록"
                nn.init.orthogonal_(param)
            else:
                # 나머지 가중치 → Xavier 초기화 (가우시안과 유사)
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            # 바이어스 → 0으로 초기화
            nn.init.zeros_(param)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {total_params:,}")
    return model


def train_step(model, batch_src, batch_src_len, batch_tgt,
               optimizer, criterion, clip=1.0):
    """
    한 미니배치 학습

    Appendix B.2 학습 전략:
    - 미니배치 크기: 논문은 80문장 (여기서는 배치 크기에 따라 자동)
    - 기울기 클리핑(Gradient Clipping): norm > 1이면 1로 스케일다운
      "차의 속도가 너무 빨라지면 자동으로 브레이크"
    - 옵티마이저: 논문은 Adadelta, 여기서는 Adam으로 근사 가능
    """
    model.train()
    optimizer.zero_grad()

    # 순전파
    outputs, _ = model(batch_src, batch_src_len, batch_tgt,
                       teacher_forcing_ratio=0.5)
    # outputs: (tgt_len-1, batch, vocab_size)
    # 목표: tgt의 1번째 ~ 마지막 토큰 (SOS 제외)

    tgt_len   = outputs.shape[0]
    batch_size = outputs.shape[1]

    # 손실 계산: 각 스텝의 예측과 정답 비교
    loss = criterion(
        outputs.reshape(tgt_len * batch_size, -1),
        batch_tgt[:, 1:].reshape(-1)   # SOS 토큰 제외
    )

    # 역전파
    loss.backward()

    # 기울기 클리핑 (Appendix B.2)
    # ‖gradient‖ > clip → gradient = gradient / ‖gradient‖
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    return loss.item()


# ──────────────────────────────────────────────────────────────
# 6. 추론: 번역 + 어텐션 시각화
# ──────────────────────────────────────────────────────────────
def translate(model, src_tokens, src_vocab, tgt_vocab,
              max_len=50, device='cpu'):
    """
    학습된 모델로 번역 수행

    Beam Search는 생략하고 Greedy Decoding으로 구현
    (논문은 Beam Search 사용, Section 4.2)

    Returns:
        translated: 번역된 단어 리스트
        attentions: 어텐션 행렬 (논문 Figure 3처럼 시각화 가능)
    """
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)
        src_len = torch.tensor([len(src_tokens)]).to(device)

        encoder_outputs, encoder_hidden = model.encoder(src, src_len)
        mask = model.create_mask(src)
        encoder_outputs_transformed = model.decoder.attention.Ua(encoder_outputs)
        decoder_hidden = torch.tanh(model.decoder.fc_hidden(encoder_hidden))

        # <SOS> 토큰으로 시작
        input_token = torch.tensor([tgt_vocab['<sos>']]).to(device)

        translated = []
        attentions = []

        for _ in range(max_len):
            pred, decoder_hidden, alpha = model.decoder(
                input_token, decoder_hidden,
                encoder_outputs, encoder_outputs_transformed, mask
            )

            # 가장 높은 확률의 단어 선택 (Greedy)
            next_token = pred.argmax(1)
            attentions.append(alpha.squeeze(0).cpu())

            word = list(tgt_vocab.keys())[next_token.item()]
            if word == '<eos>':
                break
            translated.append(word)
            input_token = next_token

        # attentions: (번역 길이, 원문 길이)
        # → 논문 Figure 3처럼 plt.imshow()로 시각화 가능
        attention_matrix = torch.stack(attentions)
        return translated, attention_matrix


def visualize_attention(attention_matrix, src_words, tgt_words):
    """
    논문 Figure 3 스타일의 어텐션 정렬 시각화

    사용법:
        import matplotlib.pyplot as plt
        translated, attn = translate(model, ...)
        visualize_attention(attn, src_words, translated)

    시각화 해석:
        - 밝은 픽셀 = αᵢⱼ가 높음 = 해당 번역 단어가 해당 원문 단어에 집중
        - 대각선 패턴 = 어순이 비슷한 언어쌍
        - 비대각선 패턴 = 형용사-명사 순서 역전 등 비단조적 정렬
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(attention_matrix.numpy(), cmap='gray', aspect='auto')

        ax.set_xticks(range(len(src_words)))
        ax.set_yticks(range(len(tgt_words)))
        ax.set_xticklabels(src_words, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(tgt_words, fontsize=10)

        ax.set_xlabel("Source (원문)", fontsize=12)
        ax.set_ylabel("Target (번역)", fontsize=12)
        ax.set_title("Attention Alignment (논문 Figure 3 스타일)\n"
                     "밝을수록 αᵢⱼ가 높음 = 해당 원문 단어에 집중", fontsize=12)

        plt.tight_layout()
        plt.savefig("attention_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("어텐션 시각화 저장: attention_visualization.png")
    except ImportError:
        print("matplotlib 없음. pip install matplotlib 후 시각화 가능")


# ──────────────────────────────────────────────────────────────
# 7. 동작 확인용 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Bahdanau Attention 모델 동작 테스트")
    print("논문: Neural Machine Translation by")
    print("      Jointly Learning to Align and Translate")
    print("      (Bahdanau et al., ICLR 2015)")
    print("=" * 60)

    torch.manual_seed(42)

    # 간단한 더미 데이터로 동작 확인
    SRC_VOCAB = 1000   # 원문 어휘 크기 (논문: 30,000)
    TGT_VOCAB = 1000   # 번역 어휘 크기 (논문: 30,000)
    BATCH     = 4
    SRC_LEN   = 12
    TGT_LEN   = 10

    # 모델 생성
    # 논문 하이퍼파라미터 (Appendix A.2.3):
    #   hidden=1000, embed=620, maxout_l=500
    # 여기서는 테스트용으로 축소
    model = build_model(SRC_VOCAB, TGT_VOCAB, embed_dim=128, hidden_size=256)

    # 더미 배치 생성
    src     = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
    src_len = torch.tensor([SRC_LEN] * BATCH)
    tgt     = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))

    # 순전파 테스트
    outputs, attentions = model(src, src_len, tgt, teacher_forcing_ratio=0.5)

    print(f"\n[Shape 확인]")
    print(f"  입력 (원문):            {src.shape}        → (batch, src_len)")
    print(f"  입력 (번역 정답):       {tgt.shape}        → (batch, tgt_len)")
    print(f"  출력 (단어 예측):       {outputs.shape}  → (tgt_len-1, batch, vocab)")
    print(f"  어텐션 가중치 αᵢⱼ:    {attentions.shape}  → (tgt_len-1, batch, src_len)")

    # 어텐션 가중치 합 확인 (각 스텝에서 합 = 1이어야 함)
    attn_sum = attentions[0].sum(dim=1)
    print(f"\n[어텐션 가중치 검증]")
    print(f"  첫 스텝 αᵢⱼ 합 (≈1.0): {attn_sum.tolist()}")

    # 간단한 학습 루프 테스트
    print(f"\n[학습 루프 테스트 (5 스텝)]")
    # 논문 Appendix B.2: Adadelta 사용 (여기서는 Adam으로 근사)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 무시

    for step in range(5):
        loss = train_step(model, src, src_len, tgt, optimizer, criterion)
        print(f"  Step {step+1}: loss = {loss:.4f}")

    print("\n모든 테스트 통과!")
    print("\n[추가 실험을 위한 권장 설정]")
    print("  실제 번역 학습 시:")
    print("    - 데이터: WMT 영어-프랑스어 병렬 코퍼스")
    print("    - 어휘: BPE(Byte Pair Encoding) 적용 후 30,000 토큰")
    print("    - 옵티마이저: Adadelta (ε=1e-6, ρ=0.95)")
    print("    - 학습 시간: GPU 기준 약 5일 (논문 기준)")
    print("    - 평가: BLEU score (논문 Table 1 참조)")
