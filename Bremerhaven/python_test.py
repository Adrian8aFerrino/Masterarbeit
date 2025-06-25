import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


# PROJECT XI: CSV creator mess
def convert_csv_to_european_format():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not file_path:
        print("No file selected.")
        return

    try:
        df = pd.read_csv(file_path, delimiter='\t', dtype=str)
        df = df.drop(columns=["Day-of-Year Demand Index"])
        string_cols = ['Symbol', 'Company', 'Date']
        numeric_cols = [col for col in df.columns if col not in string_cols]

        for col in numeric_cols:
            df[col] = df[col].str.replace('"', '', regex=False)
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df["Crude Inventory Level "] = (1000 + np.arange(len(df["Date"])) +
                                        df["Date"].index.to_series().ewm(span=5).mean())
        df["Crude Inventory Level "] = df["Crude Inventory Level "].apply(lambda x: f"{x:.2f}".replace('.', ','))

        # Convert numeric columns back to string with European decimal commas
        for col in numeric_cols:
            df[col] = df[col].map(lambda x: f"{x:.6f}".replace('.', ',') if pd.notnull(x) else '')

        dir_name = os.path.dirname(file_path)
        new_file_path = os.path.join(dir_name, f"streamlit_testing_data.csv")

        df.to_csv(new_file_path, sep=';', index=False)

    except Exception as e:
        print(f"Error processing file: {e}")


# Example usage
if __name__ == "__main__":
    convert_csv_to_european_format()
