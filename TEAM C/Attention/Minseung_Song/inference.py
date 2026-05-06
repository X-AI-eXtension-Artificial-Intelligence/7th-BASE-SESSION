"""학습된 attention 모델을 불러와서 직접 프랑스어 -> 영어 번역을 테스트하는 스크립트.
(load_data.py에서 reverse=True로 설정되어 있어서 프랑스어가 입력, 영어가 출력이다.)
사용법: python inference.py
실행하면 프랑스어 문장을 입력받고, 모델이 영어로 번역한 결과를 출력한다.
'quit' 입력 시 종료.
"""
import torch
import pickle
import re
import unicodedata
import numpy as np
import model as model_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1


def unicodeToAscii(s):
    """유니코드를 ASCII로 변환. load_data.py와 동일한 전처리."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """소문자 변환, 특수문자 처리. load_data.py와 동일한 전처리.
    학습 때 사용한 전처리와 똑같이 해야 단어 사전에서 매칭이 된다."""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def translate(sentence, enc_dec, input_lang, output_lang):
    """영어 문장을 입력받아 프랑스어로 번역하는 함수.

    흐름:
    1. 입력 문장을 전처리 (소문자, 특수문자 정리)
    2. 단어 -> 숫자 인덱스로 변환 (학습 때 만든 단어 사전 사용)
    3. 모델에 넣어서 출력 인덱스 시퀀스 생성
    4. 출력 인덱스를 다시 프랑스어 단어로 변환
    """
    with torch.no_grad():
        # 전처리
        normalized = normalizeString(sentence)
        words = normalized.split(' ')

        # 단어 사전에 없는 단어 체크
        unknown_words = [w for w in words if w not in input_lang.word2index]
        if unknown_words:
            print(f"  [주의] 학습 데이터에 없는 단어: {unknown_words}")
            print(f"  학습 데이터가 간단한 문장 위주라서, 복잡한 단어는 인식 못할 수 있다.")

        # 단어 -> 인덱스 변환
        # 사전에 없는 단어는 건너뛴다
        input_ids = []
        for w in words:
            if w in input_lang.word2index:
                input_ids.append(input_lang.word2index[w])

        if not input_ids:
            print("  인식 가능한 단어가 없다.")
            return

        # 텐서 변환 (배치 차원 추가)
        input_tensor = torch.zeros(1, MAX_LENGTH, dtype=torch.long, device=device)
        input_mask = torch.zeros(1, MAX_LENGTH, dtype=torch.long, device=device)

        for i, idx in enumerate(input_ids[:MAX_LENGTH]):
            input_tensor[0][i] = idx
            input_mask[0][i] = 1

        # 모델 추론 (target_tensor 없이 호출하면 greedy decoding)
        decoder_outputs, _ = enc_dec(input_tensor, input_mask)
        topv, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze().cpu().numpy()

        # 인덱스 -> 단어 변환
        output_words = []
        for idx in decoded_ids:
            word = output_lang.index2word[idx]
            if word == 'EOS':
                break
            if word == 'SOS':
                continue
            output_words.append(word)

        return ' '.join(output_words)


def main():
    # 저장된 모델과 언어 데이터 불러오기
    print("모델 로딩 중...")

    # 언어 데이터 (단어 사전) 로드
    with open('lang_data.pkl', 'rb') as f:
        lang_data = pickle.load(f)
    input_lang = lang_data['input_lang']
    output_lang = lang_data['output_lang']

    # 모델 로드
    checkpoint = torch.load('attention_model.pt', map_location=device, weights_only=True)
    enc_dec = model_module.EncoderDecoder(
        checkpoint['hidden_size'],
        checkpoint['input_vocab_size'],
        checkpoint['output_vocab_size']
    ).to(device)
    enc_dec.load_state_dict(checkpoint['model_state_dict'])
    enc_dec.eval()  # 평가 모드 (dropout 등 비활성화)

    print("모델 로딩 완료!")
    print(f"입력 언어: {input_lang.name} ({input_lang.n_words} 단어)")
    print(f"출력 언어: {output_lang.name} ({output_lang.n_words} 단어)")
    print("-" * 50)
    print("프랑스어 문장을 입력하면 영어로 번역합니다.")
    print("이 모델은 간단한 문장(10단어 이하)으로 학습되었습니다.")
    print("예시: je suis heureux / il est professeur / elle est ici")
    print("종료하려면 'quit' 입력")
    print("-" * 50)

    while True:
        sentence = input("\n[French] >> ").strip()
        if sentence.lower() == 'quit':
            print("종료합니다.")
            break
        if not sentence:
            continue

        result = translate(sentence, enc_dec, input_lang, output_lang)
        if result:
            print(f"[English] >> {result}")


if __name__ == '__main__':
    main()
