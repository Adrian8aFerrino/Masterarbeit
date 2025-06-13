import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('unique_data_graphs/bdew_germany_gas.xlsx')


def plot_stacked_bar_chart(df):
    df['Monate/Jahr'] = df['Monate/Jahr'].dt.strftime('%b-%Y')
    df.set_index('Monate/Jahr', inplace=True)
    ax = df.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title("Herkunft der in Deutschland zwischen 2022 und 2023 verwendeten Erdgaseinfuhren")
    plt.xlabel('Monate/Jahr')
    plt.ylabel('Prozent (%)')
    plt.legend(title='Herkunft', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


# plot_stacked_bar_chart(df)

connections = [
    ('Supply', 'Bus'),
    ('Bus', 'Sinks'),
    ('Sinks', 'Industry'),
    ('Sinks', 'Transport'),
    ('Sinks', 'Residential'),
    ('Sinks', 'Commercial'),
    ('Sinks', 'Agriculture'),
]


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

positions = {
    'Gesamtstrombedarf': (0.5, 0.55),
    'Industrie': (0.1, 0.4),
    'Verkehr': (0.3, 0.4),
    'Haushalt': (0.5, 0.4),
    'Wirtschaft': (0.7, 0.4),
    'Land- und Forstwirtschaft': (0.9, 0.4),
}

connections = [
    ('Gesamtstrombedarf', 'Industrie'),
    ('Gesamtstrombedarf', 'Verkehr'),
    ('Gesamtstrombedarf', 'Haushalt'),
    ('Gesamtstrombedarf', 'Wirtschaft'),
    ('Gesamtstrombedarf', 'Land- und Forstwirtschaft'),
]

for start, end in connections:
    start_x, start_y = positions[start]
    end_x, end_y = positions[end]
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', lw=1, color='black'), zorder=0)

for node, (x, y) in positions.items():
    if node in ['Industrie', 'Verkehr', 'Haushalt', 'Wirtschaft', 'Land- und Forstwirtschaft']:
        width, height = 0.14, 0.06
        bbox = FancyBboxPatch((x - width/2, y - height/2),
                              width, height,
                              boxstyle="Round4,pad=0.02",
                              mutation_aspect=1.5,
                              facecolor='#0033CC',
                              edgecolor='black',
                              zorder=1)
        ax.add_patch(bbox)
        ax.text(x, y, node, ha='center', va='center', color='white', weight='bold', zorder=2)
    else:
        width, height = 0.14, 0.06
        bbox = FancyBboxPatch((x - width / 2, y - height / 2),
                              width, height,
                              boxstyle="Round4,pad=0.02",
                              mutation_aspect=1.5,
                              facecolor='lightblue',
                              edgecolor='black',
                              zorder=1)
        ax.add_patch(bbox)
        ax.text(x, y, node, ha='center', va='center', color='black', weight='bold', zorder=2)

details = {
    'Industrie': 'De-Industrialisierung \nElektrifizierung von Prozessen',
    'Verkehr': 'E-Mobilität \nLadeinfrastruktur \nBatterietechnologie',
    'Haushalt': 'Elektrifizierung von Heizungssystemen',
    'Wirtschaft': 'Digitalisierung \n KI Steigerung',
    'Land- und Forstwirtschaft': 'Elektrifizierung \n EU-Landwirtschaftspolitik'
}

for node, text in details.items():
    x, y = positions[node]
    ax.text(x, y - 0.08, text, ha='center', va='top', fontsize=8, color='black', zorder=2)

plt.tight_layout()
plt.show()

