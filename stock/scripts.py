import pandas as pd
import os
from pathlib import Path

def merge_csv_files():
    stock_data_path = Path("2")
    phase2_path = Path("CG2408_PHASE4")
    output_path = Path("merged_stock data")
    
    output_path.mkdir(exist_ok=True)
    
    stock_files = {f.stem: f for f in stock_data_path.glob("*.csv")}
    
    phase2_files = {f.stem: f for f in phase2_path.glob("*.csv")}
    
    for filename in stock_files.keys():
        if filename in phase2_files:
            phase2_df = pd.read_csv(phase2_files[filename])
            stock_df = pd.read_csv(stock_files[filename])
            
            merged_df = pd.concat([phase2_df, stock_df], ignore_index=True)
            
            output_file = output_path / f"{filename}.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"已合并并保存文件: {filename}.csv")

if __name__ == "__main__":
    merge_csv_files()
