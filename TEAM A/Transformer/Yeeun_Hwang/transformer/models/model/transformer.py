import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """
    원본 논문 "Attention Is All You Need"에서 제안된 Transformer 모델의 전체 구조.
    Encoder와 Decoder를 결합하여 seq2seq(시퀀스-투-시퀀스) 작업을 수행한다.
    주로 기계 번역, 텍스트 요약 등의 NLP 태스크에 사용된다.
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        """
        Transformer 모델 초기화
            src_pad_idx  : 소스(입력) 시퀀스의 패딩 토큰 인덱스. 마스킹 시 패딩 위치를 무시하기 위해 사용.
            trg_pad_idx  : 타겟(출력) 시퀀스의 패딩 토큰 인덱스.
            trg_sos_idx  : 타겟 시퀀스의 시작 토큰(Start of Sentence) 인덱스. 디코더 입력의 첫 토큰으로 사용.
            d_model      : 모델 전체에서 사용되는 임베딩 및 hidden state의 차원 수 (논문 기본값: 512).
            n_head       : Multi-Head Attention에서 사용할 헤드(head)의 수. d_model은 n_head로 나누어져야 함.
            max_len      : 입력 시퀀스의 최대 길이. Positional Encoding 생성에 사용됨.
            ffn_hidden   : Feed-Forward Network(FFN) 내부 은닉층의 차원 수 (논문 기본값: 2048).
            n_layers     : Encoder와 Decoder 각각에 쌓을 레이어(블록)의 수 (논문 기본값: 6).
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 인코더: 소스 시퀀스를 받아 문맥 정보가 담긴 연속 표현(enc_src)으로 변환
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # 디코더: 타겟 시퀀스와 인코더 출력을 받아 다음 토큰에 대한 확률 분포를 출력
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        """
        Transformer의 순전파(forward pass)
            src : 소스 시퀀스 텐서. shape = (batch_size, src_len)
            trg : 타겟 시퀀스 텐서. shape = (batch_size, trg_len)
                  학습 시에는 정답 시퀀스를 한 칸씩 오른쪽으로 shift한 값이 입력됨 (teacher forcing).

        Returns:
            output : 각 타겟 위치에서의 어휘 확률 분포. shape = (batch_size, trg_len, dec_voc_size)
        """
        # 소스 패딩 마스크 생성: 인코더에서 패딩 토큰을 어텐션 계산에서 제외시키기 위해 사용
        src_mask = self.make_src_mask(src)

        # 타겟 마스크 생성: 패딩 마스크 + 미래 토큰 참조 방지(look-ahead mask)를 결합
        trg_mask = self.make_trg_mask(trg)

        # 인코더 실행: 소스 시퀀스를 문맥 벡터로 인코딩
        enc_src = self.encoder(src, src_mask)

        # 디코더 실행: 타겟 시퀀스, 인코더 출력, 마스크를 입력받아 최종 출력 생성
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        """
        소스 시퀀스에 대한 패딩 마스크 생성.
        패딩 토큰(src_pad_idx)인 위치는 False, 실제 토큰인 위치는 True로 표시한다.
        이를 통해 어텐션 계산 시 패딩 위치의 영향을 제거
        """
        # src_pad_idx가 아닌 위치를 True로 설정한 후, 어텐션 연산과의 브로드캐스팅을 위해 차원 확장
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # 결과 shape: (batch_size, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        """
        타겟 시퀀스에 대한 결합 마스크(패딩 마스크 + look-ahead 마스크) 생성.

        1. trg_pad_mask  : 타겟 시퀀스의 패딩 토큰 위치를 마스킹 (소스 마스크와 동일한 원리)
        2. trg_sub_mask  : 하삼각 행렬(lower triangular matrix)을 이용한 look-ahead 마스크.
                           디코더가 현재 위치 이후(미래)의 토큰을 참조하지 못하도록 막아
                           자기회귀(auto-regressive) 속성을 보장한다.

        두 마스크를 AND 연산으로 결합하여 패딩이면서 동시에 현재 이전 토큰만 참조 가능하게 한다.
        """
        # 1. 패딩 마스크: 패딩이 아닌 위치를 True로 표시, 차원 확장
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # 결과 shape: (batch_size, 1, trg_len, 1)

        trg_len = trg.shape[1]

        # 2. look-ahead 마스크 (subsequent mask): trg_len x trg_len 하삼각 행렬 생성
        #    예) trg_len=4일 때:
        #    [[1, 0, 0, 0],
        #     [1, 1, 0, 0],
        #     [1, 1, 1, 0],
        #     [1, 1, 1, 1]]
        #    → i번째 토큰은 0~i번째 토큰(자기 자신 포함 이전 토큰)만 볼 수 있음
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)

        # 3. 두 마스크를 AND 연산으로 결합
        #    trg_pad_mask: (batch_size, 1, trg_len, 1)
        #    trg_sub_mask: (trg_len, trg_len)
        #    브로드캐스팅 후 결과 shape: (batch_size, 1, trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask