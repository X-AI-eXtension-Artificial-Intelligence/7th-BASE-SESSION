import time

def epoch_time(start_time, end_time):                               # 에폭 시간 계산 함수
    elapsed_time = end_time - start_time                            # 시작 시간과 끝 시간의 차이를 계산하여 경과 시간 구하기
    elapsed_mins = int(elapsed_time / 60)                           # 경과 시간을 분 단위로 변환하여 정수로 반환
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))          # 경과 시간에서 분 단위로 변환한 시간을 빼서 초 단위로 반환
    return elapsed_mins, elapsed_secs
