# NLP 프로젝트 - T5 모델을 이용한 NLP 문서 요약 작업

## 1. 프로젝트 목적

-   초기에는 문해력 향상을 위한 애플리케이션을 개발하는 것이 목표
-   이전에 자연어 처리 프로젝트 경험이 있었으나 1년 이상의 AI 공부 중단으로 NLP 이해도 부족
-   현대 LLM 서비스들의 기반인 트랜스포머 모델과 NLP 태스크의 파이프라인 이해를 위해 T5 모델로 문서 요약 작업 구현

## 2. 사용 데이터셋/모델

### 데이터셋: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97) - 문서요약 텍스트

<img width="400" alt="스크린샷 2025-04-27 오후 4 21 35" src="https://github.com/user-attachments/assets/a6e13dc9-b03e-4708-a21f-21174e4c1277" />
<br />
<br />

-   법률문서, 사설잡지, 신문기사 총 3종으로 구성
-   Training, Validation 데이터 구분
-   총 389,535개 데이터 포함(법률문서: 51,662개, 사설잡지: 63,768개, 신문기사: 274,105개)
-   **이 중, 법률문서만 진행**

### 모델: [Hugging Face](https://huggingface.co/paust/pko-t5-small) - paust/pko-t5-small

<img width="450" alt="스크린샷 2025-04-27 오후 6 01 25" src="https://github.com/user-attachments/assets/9117f3b3-e338-41f9-b0bb-659f3f3ce403" />
<br />
<br />

-   **pko-t5-small(77M)** 사용
-   문서요약 태스크를 구현하려면 파인튜닝 필수
-   10개, 20개 데이터를 가지고 문서요약 평가지표로 결정한 ROUGE 수치 결과는 아래와 같음. 문서요약 사전학습 안 됨

<img width="400" alt="스크린샷 2025-04-27 오후 6 12 08" src="https://github.com/user-attachments/assets/6c00c451-3210-40e8-9765-34a7017644bf" />
<br />
<br />

## 3. 문서 요약 파이프라인 이전 데이터셋 파일 재구성

-   \[기존\] 현재 Training, Validation 디렉토리로 구분되어 있고, 각각 법률문서, 사설잡지, 신문기사 하위디렉토리를 구성

<img width="250" alt="스크린샷 2025-04-27 오후 5 02 22" src="https://github.com/user-attachments/assets/486453d8-622f-4158-a1c3-b2b09a1215e9" />
<br />
<br />

-   \[재구성\] 법률문서(law), 사설잡지(magazine), 신문기사(news)를 구분하고 각 도메인 별로 train(훈련), valid(검증), test(테스트) 데이터를 6:2:2 비율로 분리해서 JSONL 파일로 변경
-   기존 데이터셋 파일에서 원문과 생성 요약문만 추출하여 재구성. 추출 요약문(원문의 핵심 내용 3줄)은 우선 제외

<img width="150" alt="스크린샷 2025-04-27 오후 5 10 49" src="https://github.com/user-attachments/assets/752f1e68-c4b6-4af5-b7ac-160bfdc5a508" />
<br />
<br />

## 4. 문서 요약 파이프라인

1. 모델 및 토크나이저 준비
2. 데이터셋 준비
3. 최적의 하이퍼파라미터 탐색(Random Search)
4. 최적의 하이퍼파라미터로 모델 훈련
5. 모델 저장

### 1. 모델 및 토크나이저 준비

-   paust/pko-t5-base 설명에서 T5TokenizerFase 토크나이저, T5ForConditionalGeneration 모델 사용 권고
-   'paust/pko-t5-small' 모델 파라미터로 토크나이저, 모델 사전학습

### 2. 데이터셋 준비

-   전체 데이터셋 로드 방식이 아닌 스트리밍 방식으로 모델 훈련 때만 데이터 로드를 함으로써 메모리 리소스 절약
-   토큰화 처리 데이터셋 클래스(실제 이름: Vectorizer)를 만들어서 Trainer가 훈련 시 토큰화된 데이터를 로드하도록 진행
-   Train, Valid 데이터만 로드

### 3. 최적의 하이퍼파라미터 탐색

-   **ray** 를 이용하여 GPU 8개, GPU 한 개당 CPU 8코어 환경에서 실행하도록 설정
-   ROUGE 평가지표 중 eval_rougeLsum 의 최대값을 기준으로 ASHAScheduler 등록

### 4. 최적의 하이퍼파라미터로 모델 훈련

-   기존에 설정했던 하이퍼파라미터에서 최적의 하이퍼파라미터와 대응하는 부분을 업데이트
-   실제로 모델 훈련

### 5. 모델 저장

-   훈련을 마쳤으면 모델을 구성하는 내용들 백업

## 5. 파인튜닝한 모델의 ROUGE 평가지표 확인

-   스케줄러(뉴론 SLURM - amd_a100nv_8)에 학습 대기 중
