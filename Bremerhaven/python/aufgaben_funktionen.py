import random
import tkinter as tk

goals_2025 = ["Masterarbeit", "AEVO", "Job-Suche", "Website"]
tv_2025 = ["Netflix", "Prime", "Disney+", "Youtube", "Kindle"]

root = tk.Tk()
root.title("Tasks modus")
root.geometry("400x300")
initial_shows = "\n".join(random.choices(tv_2025, k=10))
initial_goals = "\n".join(random.choices(goals_2025, k=1))
label = tk.Label(root, text=f"What to watch: \n{initial_shows} \n \n What to work on: \n{initial_goals}",
                 font=("Times New Roman", 14))
label.pack(pady=10, fill="both", expand=True)
root.mainloop()
