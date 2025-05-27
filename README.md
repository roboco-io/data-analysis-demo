# 주식 시장 데이터 분석 (Stock Market Data Analysis)

주식 시장 데이터의 탐색적 데이터 분석(EDA)을 수행하는 프로젝트입니다. 이 프로젝트는 바이브 코딩(Vibe Coding)의 가능성을 보여주기 위한 데모 프로젝트로, AI를 활용한 데이터 분석 워크플로우를 시연합니다.

## 프로젝트 개요

이 프로젝트는 AI 기반 데이터 분석 솔루션의 가능성을 보여주기 위한 데모입니다. NASDAQ에 상장된 기업들의 주가 데이터를 분석하여 주요 ETF 및 주식의 성과, 변동성, 상관관계 등을 파악하고 시각화하는 과정에서 AI가 어떻게 활용될 수 있는지 보여줍니다. 데이터 분석은 Python의 데이터 분석 라이브러리와 SQL을 활용하여 수행되며, 데이터 탐색부터 인사이트 도출까지 AI가 어떻게 기여할 수 있는지 시연합니다.

## 데이터셋

- **출처**: Kaggle "Stock Market Dataset"
- **범위**: NASDAQ 상장 주식 및 ETF의 역사적 일일 가격 정보 (~2020년 4월 1일까지)
- **포맷**: CSV 파일 (주식/ETF별 개별 파일)
- **주요 필드**: Date, Open, High, Low, Close, Adj Close, Volume
- **메타데이터**: symbols_valid_meta.csv (총 8,049개 종목 정보)

## 프로젝트 구조

```
.
├── data/                      # 원본 데이터셋 (압축 파일)
├── docs/                      # 문서 및 계획
│   ├── dataset.md             # 데이터셋 설명
│   ├── stock_market_eda_plan.md # EDA 분석 계획
│   ├── eda_report.md          # 분석 결과 보고서
│   └── ideation.md            # 프로젝트 기획 및 아이디어
├── extracted_data/            # 압축 해제된 데이터
│   ├── etfs/                  # ETF 데이터 (CSV 파일)
│   ├── stocks/                # 주식 데이터 (CSV 파일)
│   └── symbols_valid_meta.csv # 메타데이터
├── notebooks/                 # 분석 스크립트
│   ├── stock_market_eda.py    # 주요 분석 스크립트 (시각화 포함)
│   └── stock_market_eda_basic.py # 기본 통계 분석 스크립트
├── CLAUDE.md                  # Claude Code 가이드라인
└── README.md                  # 프로젝트 설명
```

## 시작하기

### 요구사항

- Python 3.7 이상
- pandas, numpy, matplotlib, seaborn, plotly, scipy

### 데이터 준비

#### 1. Kaggle에서 데이터셋 다운로드하기

이 프로젝트를 실행하기 위해 다음 Kaggle 데이터셋을 다운로드해야 합니다:
- [nitindatta/finance-data](https://www.kaggle.com/datasets/nitindatta/finance-data)
- [jacksoncrow/stock-market-dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

다음 명령어를 사용하여 데이터셋을 다운로드 합니다:

```bash
curl -L -o data/finance-data.zip \
  "https://www.kaggle.com/api/v1/datasets/download/nitindatta/finance-data"
curl -L -o data/stock-market-dataset.zip \
  "https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset"
```

#### 2. 데이터 추출하기

```bash
# 데이터 압축 해제
unzip -o data/stock-market-dataset.zip -d extracted_data/
```


### 분석 실행

```bash
# 기본 분석 스크립트 실행
python notebooks/stock_market_eda_basic.py

# 전체 분석 스크립트 실행
python notebooks/stock_market_eda.py
```

## AI 활용 데모 시나리오

이 프로젝트는 다음과 같은 시나리오에서 AI의 활용 가능성을 시연합니다:

1. **데이터 탐색 및 전처리**: 
   - AI를 활용한 데이터셋 구조 파악 및 결측치/이상치 처리
   - 자동화된 데이터 품질 검사 및 정제 과정

2. **분석 코드 생성**:
   - 사용자 요구에 따른 맞춤형 분석 코드 자동 생성
   - 복잡한 데이터 시각화 코드의 즉시 생성 및 수정

3. **인사이트 도출**:
   - 데이터 패턴 자동 감지 및 해석
   - 금융 도메인 지식을 활용한 트렌드 분석 및 예측

4. **대화형 보고서 작성**:
   - 분석 결과를 바탕으로 한 자동 보고서 생성
   - 대화형 Q&A를 통한 추가 인사이트 탐색

## 주요 분석 내용

- 기본 통계 분석 (수익률, 변동성, 거래량)
- 시계열 시각화 (주가 및 수익률 트렌드)
- 수익률 분포 분석
- 자산 간 상관관계 분석
- 섹터별 성과 비교
- 주요 ETF 분석 (SPY, QQQ, IWM, VTI, GLD)

## 문서 및 리소스

이 프로젝트에는 다음과 같은 문서와 리소스가 포함되어 있습니다:

### 프로젝트 문서

- [**dataset.md**](docs/dataset.md): 사용된 데이터셋에 대한 상세 설명
- [**stock_market_eda_plan.md**](docs/stock_market_eda_plan.md): 데이터 분석 계획 및 접근 방법
- [**eda_report.md**](docs/eda_report.md): 실제 분석 결과와 발견된 인사이트
- [**ideation.md**](docs/ideation.md): 프로젝트 기획 배경 및 개발 아이디어

### 노트북 및 스크립트

- [**stock_market_eda.py**](notebooks/stock_market_eda.py): 주요 분석 스크립트
- [**stock_market_eda_basic.py**](notebooks/stock_market_eda_basic.py): 기본 통계 분석 스크립트

## 코드 스타일 가이드라인

- PEP 8 규칙 준수
- snake_case 함수 및 변수명 사용
- 문서화된 docstring 포함 (Parameters, Returns 섹션)
- 4-space 들여쓰기
- 의미 있는 변수명 사용
- 복잡한 연산에 대한 설명 주석 추가
- f-string 형식 사용
- 임포트 순서: 표준 라이브러리, 서드파티 라이브러리, 로컬 모듈

## 에러 처리

- 파일 작업에 try/except 블록 사용
- 적절한 pandas 메서드로 누락 데이터 처리
- 설명적인 오류 메시지와 함께 함수 입력 검증

## 바이브 코딩 데모 활용 가이드

이 프로젝트를 AI 활용 데모로 사용하는 방법은 다음과 같습니다:

### 1. 분석 요청 시나리오

다음과 같은 질문으로 AI의 데이터 분석 능력을 시연할 수 있습니다:

- "주요 테크 ETF의 2019년 성과를 비교해 주세요."
- "금융 위기 시기의 주식 시장 변동성을 분석해 주세요."
- "테슬라와 애플 주가의 상관관계를 파악하고 그래프로 보여주세요."
- "가장 성과가 좋은 5개 ETF를 찾아 시각화해 주세요."

### 2. 코드 생성 시나리오

AI가 다음과 같은 분석 코드를 자동으로 생성하는 과정을 시연합니다:

- "섹터별 ETF 성과를 비교하는 히트맵 코드를 작성해 주세요."
- "주가 데이터에서 이상치를 탐지하는 함수를 만들어 주세요."
- "월별 평균 수익률을 계산하여 바차트로 시각화해 주세요."

### 3. 보고서 생성 시나리오

특정 분석 결과를 토대로 보고서를 생성하는 과정을 시연합니다:

- "코로나 팬데믹 전후의 시장 변동성 분석 보고서를 작성해 주세요."
- "투자자를 위한 5개 주요 ETF 분석 리포트를 생성해 주세요."

이 데모 시나리오는 AI가 데이터 분석 프로세스를 어떻게 가속화하고, 사용자 경험을 향상시키며, 복잡한 분석 작업을 단순화할 수 있는지 보여줍니다.