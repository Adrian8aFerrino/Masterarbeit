import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans


def vergleich_graphik(tag_ziel, tag_quelle, jahr_ziel, jahr_quelle):
    tag_quelle = tag_quelle.copy()
    tag_quelle["Tag"] = pd.to_datetime(tag_quelle["Tag"])
    tag_quelle["Jahr"] = tag_quelle["Tag"].dt.year

    plt.figure(figsize=(14, 6))
    plt.plot(tag_quelle["Tag"], tag_quelle["Strombedarf"], label="SMARD Täglicher Strombedarf",
             linewidth=0.6, color="blue")
    plt.plot(tag_ziel["Tag"], tag_ziel["Strombedarf"], label="IEA Täglicher Strombedarf",
             linewidth=0.6, color="orange")

    for _, row in jahr_ziel.iterrows():
        year = int(row["Jahr"])
        avg_daily = row["Strombedarf"] / 365
        plt.hlines(
            y=avg_daily,
            xmin=pd.Timestamp(f"{year}-01-01"),
            xmax=pd.Timestamp(f"{year}-12-31"),
            colors='green',
            linestyles='--',
            label=f"Ziel {year} (Jahresdurchschnitt)"
        )

    for _, row in jahr_quelle.iterrows():
        year = int(row["Jahr"])
        avg_daily = row["Strombedarf"] / 365
        plt.hlines(
            y=avg_daily,
            xmin=pd.Timestamp(f"{year}-01-01"),
            xmax=pd.Timestamp(f"{year}-12-31"),
            colors='red',
            linestyles="--",
            label=f"SMARD {year} (Jahresdurchschnitt)"
        )

    plt.title("Vergleich zwischen Täglicher Stromverbrauch")
    plt.xlabel("Datum")
    plt.ylabel("Strombedarf [MWh]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    custom_lines = [
        Line2D([0], [0], color='blue', linestyle='-', linewidth=1, label='SMARD Täglicher Verbrauch'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=1, label='IEA Täglicher Verbrauch'),
        Line2D([0], [0], color='red', linestyle="--", linewidth=1, label='SMARD Jahresdurchschnitt'),
        Line2D([0], [0], color='green', linestyle="--", linewidth=1, label='IEA Jahresdurchschnitt'),
    ]
    plt.legend(handles=custom_lines)
    plt.show()


def plot_elbow_method(data, tech, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title(f'Elbow Method - {tech}')
    plt.xlabel('Clusters')
    plt.ylabel('Within-cluster sum of squares (WCSS)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
