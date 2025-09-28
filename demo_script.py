#!/usr/bin/env python3
"""
Demo script for FinTech Data Curator
Demonstrates usage for different stocks and cryptocurrencies
"""

import sys
import os
from datetime import datetime
import json

# Add the main module to path (adjust as needed)
sys.path.append('.')

try:
    from fintech_data_curator import FinTechDataCurator, MarketData
except ImportError:
    print("Please ensure fintech_data_curator.py is in the same directory")
    sys.exit(1)

def demo_single_stock(symbol, exchange, days=7):
    """
    Demonstrate data collection for a single stock/crypto.
    """
    print(f"\n{'='*60}")
    print(f"DEMO: {symbol} on {exchange}")
    print(f"{'='*60}")
    
    try:
        # Initialize curator
        curator = FinTechDataCurator(days_history=days)
        
        # Collect data
        print(f"Collecting {days} days of data for {symbol}...")
        dataset = curator.curate_dataset(symbol, exchange)
        
        # Save data to separate folders
        os.makedirs(os.path.join('output', 'csv'), exist_ok=True)
        os.makedirs(os.path.join('output', 'json'), exist_ok=True)
        csv_file = os.path.join('output', 'csv', f"demo_{symbol.replace('-', '_')}.csv")
        json_file = os.path.join('output', 'json', f"demo_{symbol.replace('-', '_')}.json")
        
        curator.save_to_csv(dataset, csv_file)
        curator.save_to_json(dataset, json_file)
        
        # Display results
        print(f"\nResults for {symbol}:")
        print(f"- Total data points: {len(dataset)}")
        print(f"- Date range: {dataset[0].date} to {dataset[-1].date}")
        print(f"- Files saved: {csv_file}, {json_file}")
        
        # Show sample data points
        print(f"\nSample Data Points:")
        print("-" * 50)
        
        for i, data in enumerate(dataset[-3:]):  # Show last 3 days
            print(f"Date: {data.date}")
            print(f"Close: ${data.close_price:.2f}")
            print(f"Volume: {data.volume:,}")
            print(f"Daily Return: {data.daily_return:.4f}")
            print(f"Volatility: {data.volatility:.4f}")
            print(f"RSI: {data.rsi:.2f}")
            print(f"Sentiment: {data.news_sentiment_score:.3f}")
            print(f"News Articles: {len(data.news_headlines)}")
            if data.news_headlines:
                print(f"Sample headline: {data.news_headlines[0][:60]}...")
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return False

def generate_summary_report(results):
    """
    Generate a summary report of all demo runs.
    """
    print(f"\n{'='*60}")
    print("DEMO SUMMARY REPORT")
    print(f"{'='*60}")
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Total symbols processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for symbol, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"- {symbol}: {status}")
    
    print(f"\nGenerated Files:")
    for symbol in results.keys():
        clean_symbol = symbol.replace('-', '_')
        print(f"- output/csv/demo_{clean_symbol}.csv")
        print(f"- output/json/demo_{clean_symbol}.json")

def print_banner():
    title = " FinTech Data Curator "
    subtitle = " Minimal feature dataset for next-day prediction "
    print("\n" + "┏" + "━" * 58 + "┓")
    print("┃" + title.center(58) + "┃")
    print("┃" + subtitle.center(58) + "┃")
    print("┗" + "━" * 58 + "┛")
    print("⏱  " + f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_menu():
    print("\n" + "┌" + "─" * 58 + "┐")
    print(f"│  {'Select an option:':<56}│")
    print("├" + "─" * 58 + "┤")
    print(f"│  {'1) Curate single symbol (exchange + symbol/ticker)':<56}│")
    print(f"│  {'2) Curate multiple symbols (comma-separated)':<56}│")
    print(f"│  {'3) Show input examples and feature set':<56}│")
    print(f"│  {'4) Run assignment demo (AAPL, MSFT, BTC-USD)':<56}│")
    print(f"│  {'0) Exit':<56}│")
    print("└" + "─" * 58 + "┘")


def prompt_single() -> tuple:
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│  🧾  Single Symbol Configuration".ljust(59) + "│")
    print("├" + "─" * 58 + "┤")
    exchange = input("│  Exchange (e.g., NASDAQ, NYSE, Crypto): ").strip()
    symbol = input("│  Symbol/Ticker (e.g., AAPL, MSFT, BTC-USD): ").strip()
    days_str = input("│  Days of history (default 7): ").strip()
    print("└" + "─" * 58 + "┘")
    days = 7
    if days_str.isdigit():
        days = int(days_str)
    return exchange, symbol, days


def prompt_multiple() -> tuple:
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│  🧾  Multiple Symbols Configuration".ljust(59) + "│")
    print("├" + "─" * 58 + "┤")
    exchange = input("│  Exchange for all symbols (e.g., NASDAQ, Crypto): ").strip()
    symbols = input("│  Symbols/Tickers (comma-separated): ").strip()
    symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
    days_str = input("│  Days of history (default 7): ").strip()
    print("└" + "─" * 58 + "┘")
    days = 7
    if days_str.isdigit():
        days = int(days_str)
    return exchange, symbols_list, days


def show_examples():
    print("\n" + "┌" + "─" * 58 + "┐")
    print(f"│  {'Examples & Feature Set':<56}│")
    print("├" + "─" * 58 + "┤")
    print(f"│  {'Exchanges & Symbols:':<56}│")
    print(f"│  {'- NASDAQ':<16}{'AAPL, MSFT':<40}│")
    print(f"│  {'- Crypto':<16}{'BTC-USD, ETH-USD':<40}│")
    print("├" + "─" * 58 + "┤")
    print(f"│  {'Minimal feature set per day:':<56}│")
    print(f"│  {'- Structured: open, high, low, close, volume':<56}│")
    print(f"│  {'  daily_return, volatility, sma_5, sma_20, rsi':<56}│")
    print(f"│  {'- Unstructured: news_headlines[], news_sentiment_score':<56}│")
    print(f"│  {'Outputs: output/csv & output/json':<56}│")
    print("└" + "─" * 58 + "┘")


def verify_outputs(symbols):
    print("\n" + "┌" + "─" * 58 + "┐")
    print(f"│  {'Data Verification':<56}│")
    print("├" + "─" * 58 + "┤")
    for symbol in symbols:
        clean_symbol = symbol.replace('-', '_')
        csv_file = os.path.join('output', 'csv', f"demo_{clean_symbol}.csv")
        json_file = os.path.join('output', 'json', f"demo_{clean_symbol}.json")
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            left = f"{symbol:<12}"
            mid = f"CSV rows: {len(df):<6}  JSON: {len(json_data):<6}"
            print(f"│  {left}{mid:<44}│")
            present = all(col in df.columns for col in ['open_price', 'close_price', 'volume', 'daily_return', 'rsi', 'news_sentiment_score'])
            print(f"│  {'Features present:':<20}{str(present):<36}│")
            dr = f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}"
            print(f"│  {'Date range:':<20}{dr:<36}│")
        except Exception as e:
            msg = f"Error verifying {symbol}: {str(e)}"
            print(f"│  {msg:<56}│")
    print("└" + "─" * 58 + "┘")


def run_assignment_demo():
    test_symbols = [
        ("AAPL", "NASDAQ"),
        ("MSFT", "NASDAQ"),
        ("BTC-USD", "Crypto")
    ]
    results = {}
    for symbol, exchange in test_symbols:
        success = demo_single_stock(symbol, exchange, days=7)
        results[symbol] = success
        import time
        time.sleep(1)
    generate_summary_report(results)
    verify_outputs(results.keys())


def main():
    """
    Interactive console menu for FinTech Data Curator.
    """
    print_banner()
    while True:
        print_menu()
        choice = input("Enter choice: ").strip()
        if choice == '1':
            exchange, symbol, days = prompt_single()
            results = {symbol: demo_single_stock(symbol, exchange, days)}
            generate_summary_report(results)
            verify_outputs(results.keys())
        elif choice == '2':
            exchange, symbols_list, days = prompt_multiple()
            results = {}
            for sym in symbols_list:
                success = demo_single_stock(sym, exchange, days)
                results[sym] = success
                import time
                time.sleep(1)
            generate_summary_report(results)
            verify_outputs(results.keys())
        elif choice == '3':
            show_examples()
        elif choice == '4':
            run_assignment_demo()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 0-4.")

if __name__ == "__main__":
    main()