#!/usr/bin/env python
# coding: utf-8

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
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로딩 및 전처리 함수
def load_stock_data(ticker, data_type='etfs'):
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

# 2. 기본 통계 분석 및 시각화 함수
def calculate_basic_stats(tickers, data_type='etfs'):
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

def plot_price_history(ticker, data_type='etfs', start_date=None, end_date=None):
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
    
    # 캔들스틱 차트
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{ticker} 가격 추이', '거래량'),
                        row_heights=[0.7, 0.3])
    
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
    
    return fig

def plot_returns_distribution(ticker, data_type='etfs'):
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
    return fig

# 3. 고급 분석 함수
def correlation_analysis(tickers, start_date=None, end_date=None, data_type='etfs'):
    """
    여러 종목 간 상관관계 분석
    
    Parameters:
    -----------
    tickers : list
        분석할 티커 심볼 리스트
    start_date, end_date : str
        분석 시작일과 종료일 (YYYY-MM-DD 형식)
    data_type : str
        'stocks' 또는 'etfs'
        
    Returns:
    --------
    DataFrame
        상관관계 매트릭스
    """
    # 각 종목의 수익률 데이터 수집
    returns_data = {}
    
    for ticker in tickers:
        df = load_stock_data(ticker, data_type)
        if df is not None:
            df = preprocess_data(df)
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            returns_data[ticker] = df['Daily_Return']
    
    # 수익률 데이터프레임 생성
    returns_df = pd.DataFrame(returns_data)
    
    # 상관관계 계산
    corr_matrix = returns_df.corr()
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('종목 간 수익률 상관관계')
    plt.tight_layout()
    
    return corr_matrix, plt.gcf()

def main():
    print("주식 시장 데이터 EDA 시작...")
    
    # 1. 메타데이터 확인
    meta_df = load_metadata()
    if meta_df is not None:
        print(f"메타데이터 로드 완료. 총 {len(meta_df)} 종목 정보가 있습니다.")
        print("\n메타데이터 샘플:")
        print(meta_df.head())
        print("\n메타데이터 칼럼:")
        print(meta_df.columns.tolist())
    
    # 2. ETF 데이터 확인
    etf_tickers = ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD']  # 주요 ETF 예시
    
    # 분석 대상 종목이 있는지 확인
    available_etfs = []
    for ticker in etf_tickers:
        df = load_stock_data(ticker)
        if df is not None:
            available_etfs.append(ticker)
    
    if not available_etfs:
        # 사용 가능한 ETF 찾기
        import glob
        all_etfs = [os.path.basename(x).replace('.csv', '') for x in glob.glob('../extracted_data/etfs/*.csv')]
        if all_etfs:
            print(f"기본 ETF를 찾을 수 없습니다. 대신 사용 가능한 ETF: {all_etfs[:5]}")
            available_etfs = all_etfs[:5]
        else:
            print("분석할 ETF 데이터를 찾을 수 없습니다.")
            return
    
    print(f"\n분석 대상 ETF: {available_etfs}")
    
    # 3. 기본 통계 분석
    stats_df = calculate_basic_stats(available_etfs)
    print("\n기본 통계 분석 결과:")
    print(stats_df)
    
    # 4. 첫 번째 ETF에 대한 가격 히스토리 시각화
    sample_ticker = available_etfs[0]
    print(f"\n{sample_ticker} ETF 가격 차트 생성...")
    fig = plot_price_history(sample_ticker)
    
    # 5. 수익률 분포 분석
    print(f"\n{sample_ticker} ETF 수익률 분포 분석...")
    returns_fig = plot_returns_distribution(sample_ticker)
    
    # 6. 상관관계 분석
    print("\n상관관계 분석...")
    if len(available_etfs) > 1:
        corr_matrix, corr_fig = correlation_analysis(available_etfs)
        print("\n상관관계 매트릭스:")
        print(corr_matrix)
    
    print("\nEDA 분석 완료!")
    
    # 그래프 저장
    fig.write_html('../docs/price_history.html')
    
    print("\n분석 결과 그래프가 docs 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()