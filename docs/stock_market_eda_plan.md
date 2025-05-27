# 주식 시장 데이터셋 EDA 계획

## 1. 데이터셋 개요 및 구조 파악

### 데이터셋 설명
- NASDAQ에서 거래되는 모든 주식과 ETF의 역사적 일일 가격 정보를 포함
- 데이터는 2020년 4월 1일까지의 가격 정보를 포함
- Yahoo Finance를 통해 yfinance 파이썬 패키지로 수집된 데이터

### 데이터 구조
- 주식 데이터는 `/stocks` 폴더에, ETF 데이터는 `/etfs` 폴더에 저장
- 각 파일명은 해당 종목의 티커 심볼
- 메타데이터는 `symbols_valid_meta.csv`에 저장

### 데이터 필드
- Date: 거래일
- Open: 시가
- High: 일중 최고가
- Low: 일중 최저가
- Close: 종가 (분할 조정)
- Adj Close: 조정 종가 (배당 및 분할 조정)
- Volume: 거래량

## 2. 데이터 탐색 단계

### 2.1 기본 데이터 통계 분석
- 데이터셋 크기 확인 (주식 및 ETF 종목 수)
- 각 종목별 데이터 기간 확인 (시작일, 종료일)
- 결측치 및 이상치 확인
- 기본 통계량 계산 (평균, 중앙값, 표준편차, 최소/최대값 등)
- 데이터 품질 평가 (결측치 비율, 이상치 비율)

### 2.2 시계열 데이터 시각화
- 주요 지수 및 대표 종목의 시간에 따른 가격 변화 시각화
- 거래량과 가격 관계 분석
- 일일 변동성 분석
- 수익률 분포 시각화
- 이동평균선을 활용한 추세 분석

### 2.3 섹터별/산업별 분석
- 메타데이터를 활용한 섹터/산업별 종목 분류
- 섹터별 성과 비교 분석
- 시장 상황에 따른 섹터별 반응 분석
- 섹터 간 상관관계 분석

### 2.4 상관관계 분석
- 주요 종목 간 상관관계 분석
- ETF와 구성 종목 간의 관계 분석
- 시장 지수와 개별 종목 간의 상관관계
- 히트맵을 통한 상관관계 시각화

### 2.5 시장 이벤트 영향 분석
- 주요 시장 이벤트 기간 동안의 가격 및 거래량 변화 분석
- 이벤트 전후 변동성 비교
- 이벤트에 따른 섹터별 반응 차이 분석

## 3. 구현 계획

### 3.1 데이터 로딩 및 전처리
```python
# 필요한 라이브러리 설치 및 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
from datetime import datetime

# 데이터 로딩 함수 구현
def load_stock_data(ticker, data_type='stocks'):
    """
    특정 티커의 주가 데이터를 로드
    
    Parameters:
    -----------
    ticker : str
        주식 또는 ETF 티커 심볼
    data_type : str
        'stocks' 또는 'etfs'
        
    Returns:
    --------
    DataFrame
        해당 티커의 주가 데이터
    """
    file_path = f'../extracted_data/{data_type}/{ticker}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        return df
    else:
        print(f"데이터 파일이 존재하지 않습니다: {file_path}")
        return None

# 메타데이터 로딩
def load_metadata():
    """
    종목 메타데이터 로드
    
    Returns:
    --------
    DataFrame
        모든 종목의 메타데이터
    """
    meta_path = '../extracted_data/symbols_valid_meta.csv'
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        return df
    else:
        print(f"메타데이터 파일이 존재하지 않습니다: {meta_path}")
        return None

# 데이터 전처리 및 정리
def preprocess_data(df):
    """
    주가 데이터 전처리
    
    Parameters:
    -----------
    df : DataFrame
        주가 데이터
        
    Returns:
    --------
    DataFrame
        전처리된 주가 데이터
    """
    # 결측치 확인 및 처리
    if df.isnull().sum().sum() > 0:
        print(f"결측치 개수: {df.isnull().sum().sum()}")
        df = df.fillna(method='ffill')  # 전일 데이터로 결측치 채우기
    
    # 일일 수익률 계산
    df['Daily_Return'] = df['Adj Close'].pct_change()
    
    # 변동성 계산 (20일 이동 표준편차)
    df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
    
    # 이동평균선 계산
    df['MA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['MA_200'] = df['Adj Close'].rolling(window=200).mean()
    
    return df
```

### 3.2 기본 통계 분석 및 시각화
```python
# 기본 통계량 계산
def calculate_basic_stats(tickers, data_type='stocks'):
    """
    여러 종목의 기본 통계량 계산
    
    Parameters:
    -----------
    tickers : list
        분석할 티커 심볼 리스트
    data_type : str
        'stocks' 또는 'etfs'
        
    Returns:
    --------
    DataFrame
        각 종목의 기본 통계량
    """
    results = []
    
    for ticker in tickers:
        df = load_stock_data(ticker, data_type)
        if df is not None:
            stats_dict = {
                'Ticker': ticker,
                'Start_Date': df.index.min(),
                'End_Date': df.index.max(),
                'Trading_Days': len(df),
                'Avg_Price': df['Adj Close'].mean(),
                'Min_Price': df['Adj Close'].min(),
                'Max_Price': df['Adj Close'].max(),
                'Avg_Volume': df['Volume'].mean(),
                'Volatility': df['Adj Close'].pct_change().std() * np.sqrt(252),  # 연간 변동성
                'Total_Return': (df['Adj Close'][-1] / df['Adj Close'][0]) - 1
            }
            results.append(stats_dict)
    
    return pd.DataFrame(results)

# 시계열 데이터 시각화
def plot_price_history(ticker, data_type='stocks', start_date=None, end_date=None):
    """
    특정 종목의 가격 히스토리 시각화
    
    Parameters:
    -----------
    ticker : str
        티커 심볼
    data_type : str
        'stocks' 또는 'etfs'
    start_date, end_date : str
        분석 시작일과 종료일 (YYYY-MM-DD 형식)
    """
    df = load_stock_data(ticker, data_type)
    if df is None:
        return
    
    df = preprocess_data(df)
    
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Plotly를 사용한 인터랙티브 차트
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{ticker} 가격 추이', '거래량'),
                        row_heights=[0.7, 0.3])
    
    # 캔들스틱 차트
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='가격'),
                  row=1, col=1)
    
    # 이동평균선 추가
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'],
                             line=dict(color='blue', width=1),
                             name='50일 이동평균'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'],
                             line=dict(color='red', width=1),
                             name='200일 이동평균'),
                  row=1, col=1)
    
    # 거래량 바 차트
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량'),
                  row=2, col=1)
    
    fig.update_layout(title=f'{ticker} 주가 및 거래량 분석',
                      xaxis_title='날짜',
                      yaxis_title='가격',
                      xaxis_rangeslider_visible=False,
                      height=800)
    
    fig.show()

# 분포 시각화
def plot_returns_distribution(ticker, data_type='stocks'):
    """
    수익률 분포 시각화
    
    Parameters:
    -----------
    ticker : str
        티커 심볼
    data_type : str
        'stocks' 또는 'etfs'
    """
    df = load_stock_data(ticker, data_type)
    if df is None:
        return
    
    df = preprocess_data(df)
    
    # 일일 수익률 분포
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 히스토그램
    sns.histplot(df['Daily_Return'].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f'{ticker} 일일 수익률 분포')
    axes[0].set_xlabel('일일 수익률')
    axes[0].set_ylabel('빈도')
    
    # QQ 플롯으로 정규성 확인
    stats.probplot(df['Daily_Return'].dropna(), plot=axes[1])
    axes[1].set_title('수익률 Q-Q 플롯')
    
    plt.tight_layout()
    plt.show()
```

### 3.3 고급 분석
```python
# 상관관계 분석
def correlation_analysis(tickers, start_date=None, end_date=None):
    """
    여러 종목 간 상관관계 분석
    
    Parameters:
    -----------
    tickers : list
        분석할 티커 심볼 리스트
    start_date, end_date : str
        분석 시작일과 종료일 (YYYY-MM-DD 형식)
    """
    # 각 종목의 수익률 데이터 수집
    returns_data = {}
    
    for ticker in tickers:
        # 주식과 ETF 모두 확인
        for data_type in ['stocks', 'etfs']:
            df = load_stock_data(ticker, data_type)
            if df is not None:
                df = preprocess_data(df)
                
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                returns_data[ticker] = df['Daily_Return']
                break
    
    # 수익률 데이터프레임 생성
    returns_df = pd.DataFrame(returns_data)
    
    # 상관관계 계산
    corr_matrix = returns_df.corr()
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('종목 간 수익률 상관관계')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# 섹터별 분석
def sector_analysis(start_date=None, end_date=None):
    """
    섹터별 성과 분석
    
    Parameters:
    -----------
    start_date, end_date : str
        분석 시작일과 종료일 (YYYY-MM-DD 형식)
    """
    # 메타데이터 로드
    meta_df = load_metadata()
    if meta_df is None:
        return
    
    # 섹터 정보가 있다고 가정 (실제 데이터에 따라 조정 필요)
    # 여기서는 'Security Name'에서 섹터 정보를 추출한다고 가정
    
    # 샘플 섹터 (실제 데이터에 맞게 조정 필요)
    sample_sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial']
    sector_tickers = {}
    
    # 각 섹터별 대표 종목 선정 (실제 데이터에 맞게 조정 필요)
    for sector in sample_sectors:
        sector_tickers[sector] = meta_df[meta_df['Security Name'].str.contains(sector, case=False)]['Symbol'].tolist()[:5]
    
    # 각 섹터별 성과 계산
    sector_performance = {}
    
    for sector, tickers in sector_tickers.items():
        sector_returns = []
        
        for ticker in tickers:
            df = load_stock_data(ticker, 'stocks')
            if df is not None:
                df = preprocess_data(df)
                
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # 누적 수익률 계산
                cumulative_return = (df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1
                sector_returns.append(cumulative_return)
        
        if sector_returns:
            sector_performance[sector] = np.mean(sector_returns)
    
    # 섹터별 성과 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(sector_performance.keys(), sector_performance.values())
    plt.title('섹터별 평균 수익률')
    plt.xlabel('섹터')
    plt.ylabel('평균 수익률')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return sector_performance

# 이벤트 분석
def event_analysis(ticker, event_date, window=30, data_type='stocks'):
    """
    특정 이벤트 전후 주가 변화 분석
    
    Parameters:
    -----------
    ticker : str
        티커 심볼
    event_date : str
        이벤트 발생일 (YYYY-MM-DD 형식)
    window : int
        이벤트 전후 분석 기간 (일)
    data_type : str
        'stocks' 또는 'etfs'
    """
    df = load_stock_data(ticker, data_type)
    if df is None:
        return
    
    df = preprocess_data(df)
    
    # 이벤트 날짜를 datetime으로 변환
    event_date = pd.to_datetime(event_date)
    
    # 이벤트 전후 기간 설정
    start_date = event_date - pd.Timedelta(days=window)
    end_date = event_date + pd.Timedelta(days=window)
    
    # 해당 기간 데이터 추출
    event_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    if len(event_df) == 0:
        print(f"해당 기간에 데이터가 없습니다: {start_date} ~ {end_date}")
        return
    
    # 이벤트 발생일 기준 가격을 100으로 정규화
    event_price = event_df[event_df.index.date == event_date.date()]['Adj Close'].values
    if len(event_price) == 0:
        # 정확한 이벤트 날짜가 없는 경우 가장 가까운 날짜 사용
        closest_date = event_df.index[event_df.index.searchsorted(event_date)]
        event_price = event_df.loc[closest_date, 'Adj Close']
    else:
        event_price = event_price[0]
    
    event_df['Normalized_Price'] = event_df['Adj Close'] / event_price * 100
    
    # 이벤트 전후 가격 변화 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(event_df.index, event_df['Normalized_Price'])
    plt.axvline(x=event_date, color='r', linestyle='--', label='이벤트 발생일')
    plt.title(f'{ticker} - {event_date.date()} 이벤트 전후 가격 변화')
    plt.xlabel('날짜')
    plt.ylabel('정규화된 가격 (이벤트 발생일 = 100)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 이벤트 전후 통계 계산
    pre_event = event_df[event_df.index < event_date]
    post_event = event_df[event_df.index >= event_date]
    
    pre_return = (pre_event['Adj Close'].iloc[-1] / pre_event['Adj Close'].iloc[0]) - 1 if len(pre_event) > 0 else None
    post_return = (post_event['Adj Close'].iloc[-1] / post_event['Adj Close'].iloc[0]) - 1 if len(post_event) > 0 else None
    
    pre_volatility = pre_event['Daily_Return'].std() * np.sqrt(252) if len(pre_event) > 0 else None
    post_volatility = post_event['Daily_Return'].std() * np.sqrt(252) if len(post_event) > 0 else None
    
    print(f"이벤트 전 {window}일 수익률: {pre_return:.2%}" if pre_return is not None else "이벤트 전 데이터 없음")
    print(f"이벤트 후 {window}일 수익률: {post_return:.2%}" if post_return is not None else "이벤트 후 데이터 없음")
    print(f"이벤트 전 변동성: {pre_volatility:.2%}" if pre_volatility is not None else "이벤트 전 데이터 없음")
    print(f"이벤트 후 변동성: {post_volatility:.2%}" if post_volatility is not None else "이벤트 후 데이터 없음")
```

### 3.4 인사이트 도출
```python
def generate_insights(tickers, start_date=None, end_date=None):
    """
    여러 종목에 대한 인사이트 도출
    
    Parameters:
    -----------
    tickers : list
        분석할 티커 심볼 리스트
    start_date, end_date : str
        분석 시작일과 종료일 (YYYY-MM-DD 형식)
    """
    # 기본 통계 계산
    stats_df = calculate_basic_stats(tickers)
    
    # 상위 성과 종목
    top_performers = stats_df.sort_values('Total_Return', ascending=False).head(5)
    print("=== 상위 성과 종목 ===")
    print(top_performers[['Ticker', 'Total_Return']])
    
    # 하위 성과 종목
    bottom_performers = stats_df.sort_values('Total_Return').head(5)
    print("\n=== 하위 성과 종목 ===")
    print(bottom_performers[['Ticker', 'Total_Return']])
    
    # 변동성 상위 종목
    high_volatility = stats_df.sort_values('Volatility', ascending=False).head(5)
    print("\n=== 변동성 상위 종목 ===")
    print(high_volatility[['Ticker', 'Volatility']])
    
    # 상관관계 분석
    corr_matrix = correlation_analysis(tickers, start_date, end_date)
    
    # 낮은 상관관계를 가진 종목 쌍 찾기 (분산투자 관점)
    corr_pairs = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            if tickers[i] in corr_matrix.index and tickers[j] in corr_matrix.columns:
                corr_pairs.append((tickers[i], tickers[j], corr_matrix.loc[tickers[i], tickers[j]]))
    
    low_corr_pairs = sorted(corr_pairs, key=lambda x: x[2])[:5]
    print("\n=== 낮은 상관관계를 가진 종목 쌍 (분산투자 관점) ===")
    for pair in low_corr_pairs:
        print(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
    
    # 추가 분석 방향 제안
    print("\n=== 추가 분석 방향 ===")
    print("1. 시장 상황별 성과 분석 (강세장/약세장)")
    print("2. 계절성 분석 (월별, 분기별 패턴)")
    print("3. 거시경제 지표와의 상관관계 분석")
    print("4. 기술적 지표 기반 매매 전략 백테스팅")
    print("5. 섹터 로테이션 분석")
```

## 4. 필요한 라이브러리
- pandas: 데이터 처리 및 분석
- numpy: 수치 계산
- matplotlib/seaborn: 데이터 시각화
- plotly: 인터랙티브 시각화
- scipy: 통계 분석
- statsmodels: 시계열 분석

## 5. 실행 계획
1. 데이터 로딩 및 구조 파악
   - 메타데이터 분석
   - 샘플 데이터 확인
   - 데이터 품질 평가

2. 기본 통계 분석 수행
   - 종목별 기본 통계량 계산
   - 결측치 및 이상치 처리
   - 수익률 및 변동성 계산

3. 시각화를 통한 데이터 패턴 탐색
   - 주요 지수 및 대표 종목 시각화
   - 섹터별 성과 비교 시각화
   - 수익률 분포 시각화

4. 상관관계 및 고급 분석 수행
   - 종목 간 상관관계 분석
   - 시장 이벤트 영향 분석
   - 섹터별 분석

5. 결과 정리 및 인사이트 도출
   - 주요 발견 사항 정리
   - 투자 관점에서의 인사이트
   - 추가 분석 방향 제안

## 6. 결론

이 EDA 계획을 통해 주식 시장 데이터에 대한 포괄적인 분석을 수행할 수 있습니다. 데이터의 특성을 이해하고, 시각화와 통계 분석을 통해 유의미한 패턴과 인사이트를 도출할 수 있을 것입니다. 특히 종목 간 상관관계, 섹터별 성과 차이, 시장 이벤트의 영향 등을 분석함으로써 투자 결정에 도움이 되는 정보를 얻을 수 있습니다.
