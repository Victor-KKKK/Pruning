#!/bin/bash

# 실행할 SEED 값들의 리스트
SEED_LIST="15 25 35"

# 실행할 PRUNE_AMOUNT 값들의 리스트
# 0을 0.0으로 표기하여 로그 파일 이름의 일관성을 맞춥니다.
PRUNE_LIST="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9"

echo "CIFAR10 Pruning (Seed/Prune) 실험을 시작합니다..."
echo "=================================================="

# Seed 리스트를 순회 (Outer loop)
for seed in $SEED_LIST
do
    echo "****** [SEED: $seed] 실험 시작 ******"
    
    # Prune 리스트를 순회 (Inner loop)
    for p_amount in $PRUNE_LIST
    do
        # 각 실행 결과를 저장할 로그 파일 이름 지정 (seed 포함)
        LOG_FILE="Random_RS${seed}_ResNet18_Ep100_run_prune_${p_amount}.log"

        echo "  [Seed: $seed, Prune: $p_amount] 실행 중... 로그: $LOG_FILE"
        
        # MY_SEED와 PRUNE_AMOUNT 환경 변수를 설정하여 python 스크립트 실행
        # 모든 표준 출력(stdout)과 표준 에러(stderr)를 로그 파일로 리디렉션
        MY_SEED=$seed PRUNE_AMOUNT=$p_amount python3 main.py > $LOG_FILE 2>&1
        
        echo "  [Seed: $seed, Prune: $p_amount] 실행 완료."
        echo "  ------------------------------------------------"
    done
    
    echo "****** [SEED: $seed] 실험 완료 ******"
    echo "" # 가독성을 위한 빈 줄
done

echo "=================================================="
echo "모든 (Seed/Prune) 실험이 완료되었습니다."
