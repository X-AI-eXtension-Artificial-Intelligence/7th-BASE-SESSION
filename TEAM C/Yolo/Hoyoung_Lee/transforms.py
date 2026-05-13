class ToYOLOTensor:
    """일반적인 (x, y, w, h) 타겟을 YOLO가 학습할 수 있는 (S, S, C+5) 그리드 형태로 변환"""
    
    def __call__(self, sample):
        # 1. 객체의 중심 좌표가 어느 그리드 셀(row, col)에 속하는지 계산
        # 2. 전체 이미지 기준이던 좌표를 해당 그리드 셀 안에서의 상대적 위치(0~1)로 정규화
        
        target = th.zeros((self.S, self.S, self.C + 5))
        # 해당 셀에 정보 기록: [객체 존재여부(1), 원-핫 클래스, x_norm, y_norm, w_norm, h_norm]
        target[center_row, center_col, :] = th.cat([...], dim=1)
        
        return img_tensor, target

# 이 외에도 RandomScaleTranslate(확대/축소 및 이동), RandomColorJitter(색상 왜곡), 
# RandomHorizontalFlip(좌우 반전) 등의 클래스 포함