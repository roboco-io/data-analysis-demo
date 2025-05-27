#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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

# 2. 기본 통계 분석 함수
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
            # 전처리
            df = preprocess_data(df)
            
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

def main():
    print("주식 시장 데이터 기본 EDA 시작...")
    
    # 1. 메타데이터 확인
    meta_df = load_metadata()
    if meta_df is not None:
        print(f"메타데이터 로드 완료. 총 {len(meta_df)} 종목 정보가 있습니다.")
        print("\n메타데이터 샘플:")
        print(meta_df.head())
        print("\n메타데이터 칼럼:")
        print(meta_df.columns.tolist())
        
        # ETF 카운트
        etf_count = meta_df[meta_df['ETF'] == 'Y'].shape[0]
        print(f"\nETF 개수: {etf_count}")
        
        # 거래소별 종목 수
        exchange_counts = meta_df['Listing Exchange'].value_counts()
        print("\n거래소별 종목 수:")
        print(exchange_counts)
    
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
    
    # 4. 상위 성과 및 하위 성과 ETF
    if len(stats_df) > 1:
        top_performers = stats_df.sort_values('Total_Return', ascending=False).head(3)
        bottom_performers = stats_df.sort_values('Total_Return').head(3)
        
        print("\n상위 성과 ETF:")
        print(top_performers[['Ticker', 'Total_Return']])
        
        print("\n하위 성과 ETF:")
        print(bottom_performers[['Ticker', 'Total_Return']])
    
    # 5. 개별 ETF 상세 분석
    for ticker in available_etfs[:2]:  # 처음 두 개 ETF만 상세 분석
        df = load_stock_data(ticker)
        if df is not None:
            df = preprocess_data(df)
            
            print(f"\n{ticker} ETF 상세 분석:")
            print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
            print(f"거래일 수: {len(df)}")
            
            # 월별 평균 수익률
            df['Year'] = df.index.year
            df['Month'] = df.index.month
            monthly_returns = df.groupby(['Year', 'Month'])['Daily_Return'].mean() * 100  # 백분율로 변환
            
            print("\n월별 평균 일일 수익률 (%):")
            print(monthly_returns.tail(12))  # 최근 12개월
            
            # 거래량 분석
            print(f"\n평균 일일 거래량: {df['Volume'].mean():,.0f}")
            print(f"최대 거래량: {df['Volume'].max():,.0f} ({df['Volume'].idxmax()})")
            
            # 변동성 분석
            annual_vol = df['Daily_Return'].std() * np.sqrt(252) * 100  # 연간화 및 백분율로 변환
            print(f"\n연간 변동성: {annual_vol:.2f}%")
    
    # 6. CSV 파일로 결과 저장
    stats_df.to_csv('../docs/etf_stats_summary.csv', index=False)
    print("\nEDA 분석 완료! 결과가 docs 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()