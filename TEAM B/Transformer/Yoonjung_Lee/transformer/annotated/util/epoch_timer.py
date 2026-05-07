"""
util/epoch_timer.py
- epoch 하나가 걸린 시간을 분/초 단위로 바꿔주는 간단한 보조 함수입니다.
"""


def epoch_time(start_time, end_time):
    # 초 단위 경과 시간입니다.
    elapsed_time = end_time - start_time

    # 전체 초를 분과 초로 나눕니다.
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
