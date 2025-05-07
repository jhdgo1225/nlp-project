# NLP 프로젝트 - T5 모델을 이용한 NLP 문서 요약 작업

## 1. 프로젝트 목적

-   초기에는 문해력 향상을 위한 애플리케이션을 개발하는 것이 목표
-   이전에 자연어 처리 프로젝트 경험이 있었으나 1년 이상의 AI 공부 중단으로 NLP 이해도 부족
-   현대 LLM 서비스들의 기반인 트랜스포머 모델과 NLP 태스크의 파이프라인 이해를 위해 T5 모델로 문서 요약 작업 구현

## 2. 프로젝트 목표 및 개요

-   한국어 데이터셋으로 사전학습한 T5 모델을 가지고 한국어 문서 요약 NLP 태스크 구현
-   추출적 요약(원문에서 핵심 3문장을 추출하는 방식), 추상적 요약(원문으로부터 내용이 요약된 새로운 문장을 생성하는 방식) 중 추상적 요약만 진행
-   해당 모델을 한국어 문서 요약에 특화된 모델로 개선하기 위해 AI-Hub '문서요약 텍스트' 데이터셋의 원문과 요약문을 이용한 지도 학습으로 파인튜닝

### 사용 데이터셋: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97) - 문서요약 텍스트

<img width="400" alt="스크린샷 2025-04-27 오후 4 21 35" src="https://github.com/user-attachments/assets/a6e13dc9-b03e-4708-a21f-21174e4c1277" />
<br />
<br />

-   법률문서, 사설잡지, 신문기사 총 3종으로 구성
-   Training, Validation 데이터 구분
-   총 389,535개 데이터 포함(법률문서: 51,662개, 사설잡지: 63,768개, 신문기사: 274,105개)
-   **이 중, 법률문서만 진행**

### 사용 모델: [Hugging Face](https://huggingface.co/paust/pko-t5-small) - paust/pko-t5-small

<img width="450" alt="스크린샷 2025-04-27 오후 6 01 25" src="https://github.com/user-attachments/assets/9117f3b3-e338-41f9-b0bb-659f3f3ce403" />
<br />
<br />

-   한국어 대규모 데이터셋(나무위키, 위키피디아, 모두의 말뭉치 등)을 T5의 span corruption task를 사용해서 비지도 학습 방식으로 사전학습한 paust/pko-t5-small(77M) 모델 이용
-   대상 태스크(문서 요약 작업)에 맞게 파인튜닝 진행
-   문서 요약 평가 지표로 ROUGE 사용. 테스트 데이터 10개, 20개에 대한 ROUGE 측정 결과는 아래와 같다.

<img width="400" alt="스크린샷 2025-04-27 오후 6 12 08" src="https://github.com/user-attachments/assets/6c00c451-3210-40e8-9765-34a7017644bf" />
<br />
<br />

## 4. 문서 요약 파이프라인 이전 데이터셋 파일 재구성

-   \[기존\] 현재 Training, Validation 디렉토리로 구분되어 있고, 각각 법률문서, 사설잡지, 신문기사 하위디렉토리를 구성

<img width="250" alt="스크린샷 2025-04-27 오후 5 02 22" src="https://github.com/user-attachments/assets/486453d8-622f-4158-a1c3-b2b09a1215e9" />
<br />
<br />

-   \[재구성\] 법률문서(law), 사설잡지(magazine), 신문기사(news)를 구분하고 각 도메인 별로 train(훈련), valid(검증), test(테스트) 데이터를 6:2:2 비율로 분리해서 JSONL 파일로 변경
-   기존 데이터셋 파일에서 원문과 생성 요약문만 추출하여 재구성. 추출 요약문(원문의 핵심 내용 3줄)은 우선 제외

<img width="150" alt="스크린샷 2025-04-27 오후 5 10 49" src="https://github.com/user-attachments/assets/752f1e68-c4b6-4af5-b7ac-160bfdc5a508" />
<br />
<br />

## 5. 문서 요약 파이프라인

-   각 단계별 핵심 내용 설명
-   자세한 내용 파악은 코드에서 진행

1. 모델 및 토크나이저 준비
2. 데이터셋 준비
3. 모델 훈련 준비
4. 모델 훈련
5. Trainer 체크포인트 저장, 모델•토크나이저 체크포인트 저장

### 1. 모델 및 토크나이저 준비

<br />
<br />

-   paust/pko-t5-base 설명에서 T5TokenizerFast 토크나이저, T5ForConditionalGeneration 모델 사용 권고
-   T5TokenizerFast: T5 모델 전용의 빠른 토크나이저. T5Tokenizer보다 빠른 토크나이징 수행. SentencePiece 방식으로 토큰화
-   T5ForConditionalGeneration: T5 모델 아키텍처. 조건부 텍스트 생성(질의응답, 기계번역, 문서요약 등 다양한 NLP 태스크 입력에 적합한 조건부 생성)을 위한 로직 내포
-   'paust/pko-t5-small' 체크포인트로부터 구성 및 파라미터를 T5TokenizerFast, T5ForConditionalGeneration으로 로드

### 2. 데이터셋 준비

-   데이터셋 전체를 한번에 로드하지 않고, 스트리밍 방식으로 모델을 훈련 할 때만 일부 데이터를 로드하여 학습 후 데이터를 저장하지 않음으로써 메모리 절약
-   토큰화 처리 데이터셋 클래스(실제 이름: Vectorizer)를 만들어서 Trainer가 훈련 시 토큰화된 데이터를 로드하도록 진행
-   Train, Valid 데이터만 로드

### 3. 모델 훈련 준비

-   하이퍼파라미터 및 Trainer 준비
-   하이퍼파라미터는 다음과 같이 구성함
    -   output_dir: 모델과 체크포인트 파일이 저장될 경로
    -   eval_strategy="steps": eval_steps마다 평가 수행
    -   eval_steps=200: 200 step마다 평가
    -   logging_steps=100: 훈련 중 로깅 정보를 출력할 주기를 100으로 설정
    -   save_steps=500: 500 step마다 모델 체크포인트 저장
    -   num_train_epochs=3: 학습 최대 3번
    -   per_device_train_batch_size=4: 학습 시 배치 사이즈 -> 4
    -   per_device_eval_batch_size=4: 평가 시 배치 사이즈 -> 4
    -   gradient_accumulation_steps=8: GPU 메모리 부족 시, 8번의 step에 걸쳐 gradient를 누적하여 1회 업데이트.
    -   learning_rate=3e-5: 학습률
    -   warmup_steps=500: 학습 초기 학습률을 서서히 증가시키는 단계 수
    -   fp16=torch.cuda.is_available(): GPU 사용 시, 학습 속도와 메모리 사용 최적화를 위한 float16 혼합 정밀도 학습 사용 여부
    -   predict_with_generate=True: 평가 시 generate() 메서드로 텍스트 생성
    -   generation_max_length=512: 텍스트 생성 시 최대 토큰 길이 512
    -   generation_num_beams=4: 빔 서치 디코딩에서 사용할 빔의 수 4
-   Trainer는 다음과 같이 구성함
    -   model: 학습 모델
    -   args: 학습 하이퍼파라미터
    -   train_dataset: 학습용 데이터셋
    -   eval_dataset: 평가용 데이터셋
    -   processing_class: 토크나이저 및 이미지 프로세서, 피처 익스트랙터와 같은 입력 전처리 인터페이스
    -   data_collator: 모델에 데이터를 입력하기 전 배치 단위로 데이터를 전처리하는 객체
    -   compute_metrics: 모델 평가 시 평가지표 계산을 위한 함수

### 4. 모델 훈련

-   trainer.train() 명령어 실행

### 5. trainer, 모델, 토크나이저 저장

-   훈련을 끝낸 trainer 객체의 체크포인트 저장
-   모델의 가중치, 구성(config) 저장
-   토크나이저 구성(vocab, merge, preprocessor, postprocessor pipeline, etc.), 토크나이저 기본 옵션, 특수 토큰 정보 저장
