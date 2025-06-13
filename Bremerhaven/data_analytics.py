import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_bremen = gpd.read_file("/Users/ochoa/PycharmProjects/Bremerhaven/databases/bremen_karte.geojson")

print('COLUMNS', data_bremen.columns)

# MATPLOTLIB
fig, ax = plt.subplots()
data_bremen.plot(ax=ax, column='cartodb_id', cmap='Greens', linewidth=0.8, edgecolor='black', alpha=0.8)
ax.set_title('GIS Map-test')
plt.show()

# PLOTLY
fig_1 = px.choropleth(data_bremen,
                      geojson=data_bremen.geometry,
                      locations=data_bremen.index,
                      color_continuous_scale="Greens",
                      projection="mercator")
fig_1.update_geos(fitbounds="locations")
fig_1.update_layout(title='Bremen Map using Plotly')
fig_1.show()
