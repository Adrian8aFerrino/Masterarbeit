import os
import numpy as np
import pandas as pd
from config import select_folder
from sklearn.cluster import KMeans


def bnetza_kapazitat(bnetza_path, generation_path):
    kraftwerkliste = pd.read_csv(os.path.join(bnetza_path, "bnetza_kraftwerkliste.csv"), delimiter=",",
                                 encoding="cp1252")
    tech_mapping = {
        'Abfall': 'Biomasse',
        'Biomasse': 'Biomasse',
        'Braunkohle': 'Kohle',
        'Steinkohle': 'Kohle',
        'Erdgas': 'Erdgas',
        'Grubengas': 'Erdgas',
        'Windenergie (Onshore-Anlage)': 'Onshore-Wind',
        'Windenergie (Offshore-Anlage)': 'Offshore-Wind',
        'Solare Strahlungsenergie': 'Sonne',
        'Kernenergie': 'Kernkraft'
    }

    kraftwerkliste['Technologie'] = kraftwerkliste['Auswertung Energieträger'].map(tech_mapping)
    netto_leistung = kraftwerkliste['Nettonennleistung (elektrische Wirkleistung) in MW']
    netto_leistung = (pd.to_numeric(netto_leistung, errors='coerce'))
    kraftwerkliste = kraftwerkliste[(netto_leistung.notna()) & (netto_leistung > 0)]
    kapazitat = kraftwerkliste[['Technologie', 'Nettonennleistung (elektrische Wirkleistung) in MW']].copy()
    kapazitat.iloc[:, 1] = pd.to_numeric(kapazitat.iloc[:, 1], errors='coerce')
    kapazitat.iloc[:, 1] = kapazitat.iloc[:, 1].replace(0, np.nan)
    kapazitat = kapazitat.dropna(subset=[kapazitat.columns[0], kapazitat.columns[1]])

    def assign_clusters(x):
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        return model.fit(x.values.reshape(-1, 1)).labels_

    kapazitat['Cluster'] = kapazitat.groupby(kapazitat.columns[0])[kapazitat.columns[1]].transform(assign_clusters)
    kapazitat.to_csv(os.path.join(generation_path, f"kapazitat_technologie.csv"), index=False)


def iea_energie_mix(iea_path, generation_path, country):
    energie_mix = pd.read_csv(os.path.join(iea_path, f"International Energy Agency - electricity generation sources "
                                                     f"in {country}.csv"), delimiter=",",
                              encoding="cp1252")

    generation_mapping = {
        "Coal": "Kohle",
        "Oil": "Öl",
        "Natural gas": "Erdgas",
        "Nuclear": "Kernkraft",
        "Hydro": "Wasserkraft",
        "Biofuels": "Biomasse",
        "Waste": "Biomasse",
        "Wind": "Wind",  # TODO: Needs to be split into: Onshore-Wind and Offshore-Wind (Find official source)
        "Solar PV": "Sonne",
        "Solar thermal": "Sonne",
        "Geothermal": "Geothermie",
        "Other sources": "Andere Energiequellen"
    }

    sektor_df = pd.read_csv(os.path.join(iea_path, f"International Energy Agency - electricity final consumption by "
                                                   f"sector in {country}.csv"))
    sektor_df[sektor_df.columns[0]] = (sektor_df[sektor_df.columns[0]].map(sektor_dic).
                                       fillna(sektor_df[sektor_df.columns[0]]))
    sektor_df["Value"] = sektor_df["Value"] * 277.778  # Conversion to MWh
    sektoren_spalten = sektor_df.groupby(["Year", sektor_df.columns[0]])["Value"].sum().unstack(fill_value=0)
    sektoren_spalten = sektoren_spalten.rename(columns={'Year': 'Jahr'})

    jahr_bedarf = sektor_df.groupby("Year")["Value"].sum().reset_index()
    jahr_bedarf.columns = ["Jahr", "Strombedarf"]
    jahr_bedarf_full = jahr_bedarf.merge(sektoren_spalten, left_on="Jahr", right_index=True, how="left")

    # Integrates yearly_demand dataframe as total demand
    jahr_bedarf_full = jahr_bedarf_full.merge(yearly_demand[["Jahr", "Strombedarf"]], on="Jahr", suffixes=('', '_neu'))
    jahr_bedarf_percent = jahr_bedarf_full.copy()
    jahr_bedarf_percent[list(jahr_bedarf_full.columns)[1:-1]] = (
            jahr_bedarf_percent[list(jahr_bedarf_full.columns)[1:-1]].div(jahr_bedarf_percent["Strombedarf_neu"],
                                                                          axis=0) * 100)

    sektor_pivot = sektor_df.pivot_table(values="Value", index="Year", columns=sektor_df.columns[0],
                                         aggfunc="sum", fill_value=0)
    sektor_percent = sektor_pivot.div(sektor_pivot.sum(axis=1), axis=0) * 100
    sektor_percent = sektor_percent.reset_index()
    sektor_percent = sektor_percent.rename(columns={"Year": "Jahr"})
    sektor_percent.to_csv(os.path.join(folder_path, f"stromsektor_prozent_{country}.csv"), index=False)


"""
bnetza_path = select_folder(message="Select BNETZA Folder")
generation_path = select_folder(message="Select Generation Folder")
bnetza_kapazitat(bnetza_path, generation_path)
"""
iea_path = select_folder(message="Select IEA Folder")
