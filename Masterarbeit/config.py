import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def select_folder(message):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    folder_path = filedialog.askdirectory(title=f"Select the {message} folder")
    root.destroy()
    return folder_path


def eurostat_csv_maker(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tsv"):
            tsv_path = os.path.join(folder_path, filename)
            csv_path = os.path.join(folder_path, filename.replace(".tsv", ".csv"))

            df_eurostat = pd.read_csv(tsv_path, sep='\t', dtype=str)
            df_eurostat = pd.concat([df_eurostat.iloc[:, 0].str.split(',', expand=True)
                                    .set_axis(df_eurostat.columns[0].split(','), axis=1),
                                     df_eurostat.iloc[:, 1:]], axis=1)
            df_eurostat.to_csv(csv_path, index=False)

            print(f"Converted {filename} to {os.path.basename(csv_path)}")