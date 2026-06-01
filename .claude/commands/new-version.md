# /new-version $ARGUMENTS
새 모델 버전 생성:
1. ls hMuscle/model/v* → 최신 버전 확인
2. _backup_{DATE}.py 생성
3. v{N}_integrated_full_model.py 생성
4. 변경부분: # v{N} [model-engineer]: {이유}
5. sanity check 블록 추가
