from darknet import DarkNet


def get_model(S=7, B=2, C=20):
    """
    논문 설정:
    S=7  : 이미지를 7×7 그리드로 분할
    B=2  : 각 셀당 2개의 bounding box 예측
    C=20 : PASCAL VOC 20개 클래스
    최종 출력: 7×7×30 텐서
    """
    return DarkNet(S=S, B=B, C=C)