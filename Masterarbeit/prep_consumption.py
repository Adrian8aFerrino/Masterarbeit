import os
import pandas as pd
from data_graphs import oemof_graphs
from config import select_folder


def smard_15min_csv_maker(smard_path, key_name):
    """
    The following function serves in gather all csv files that come from smard.de that represent the electricity
    consumption "Realisierter_Stromverbrauch" and electricity production "Realisierte_Erzeugung" in the 15_min format.
    This function is created due to the restriction limits for download size within the smard.de file.

    The output csv file will later be used to determine the load_curve and load_factor values for more a detailed
    electricity oemof-model configuration. The data is obtained from the following link:
    https://www.smard.de/home/downloadcenter/download-marktdaten/

    :param
        smard_path: Has to match the smard_path where the smard.de data is saved.

    :param
        key_name: Can be either "Stromverbrauch" or "Erzeugung" for electricity consumption and generation respectively.
    """
    all_df = []
    if key_name == "Stromverbrauch":
        title = f"Realisierter_{key_name}_"
    elif key_name == "Erzeugung":
        title = f"Realisierte_{key_name}_"
    else:
        print("Key_name is wrong")
        return
    print("\n Sammlung der folgenden csv-Dateien:")
    for filename in os.listdir(smard_path):
        if filename.startswith(title) and filename.endswith("_Viertelstunde.csv"):
            file_path = os.path.join(smard_path, filename)
            print(f"--- {filename} ---")
            df_smard = pd.read_csv(file_path, delimiter=";")
            df_smard["Datum von"] = pd.to_datetime(df_smard["Datum von"], format='%d.%m.%Y %H:%M')
            for col in df_smard.columns[2:]:
                df_smard[col] = (df_smard[col].astype(str).str.replace('.', '', regex=False)
                                 .str.replace(',', '.', regex=False).astype(float))
            all_df.append(df_smard)

    if not all_df:
        print("No files processed.")
        return

    all_df = pd.concat(all_df, ignore_index=True)
    if "Datum von" in all_df.columns:
        all_df["Datum von"] = pd.to_datetime(all_df["Datum von"], format='%d.%m.%Y %H:%M')
        all_df = all_df.sort_values("Datum von").reset_index(drop=True)
    all_df.to_csv(os.path.join(smard_path, f"15min_{key_name}.csv"), index=False, sep=";")
    i = 0
    for column in all_df.columns:
        print(f"{i}: {column}")
        i += 1


def smard_strombedarf(smard_path, consumption_path, storage_path, generation_path):
    """
    The following function reads the renamed smard.de file that contains the daily historical electricity consumption
    data of Germany. The smard.de data holds daily "Netzlast", "Pumpspeicher" and "Residuallast" data that is then
    saved in compact csv files for electricity demand, storage and residual electricity load at their respective
    folders.

    :param
        smard_path: : Has to match the smard_path where the smard.de data is saved.

    :return:
        jahr_bedarf: Aggregates the daily demand data from smard.de into a yearly dataframe.
    """
    tagliche_bedarf = pd.read_csv(os.path.join(smard_path, "tag_Stromverbrauch.csv"), delimiter=";")
    tagliche_bedarf["Datum von"] = pd.to_datetime(tagliche_bedarf["Datum von"], format="%d.%m.%Y")
    for col in tagliche_bedarf.columns[2:]:
        tagliche_bedarf[col] = (tagliche_bedarf[col].astype(str).str.replace('.', '', regex=False)
                                .str.replace(',', '.', regex=False).astype(float))

    tagliche_bedarf["Year"] = tagliche_bedarf["Datum von"].dt.year
    jahr_bedarf = tagliche_bedarf.groupby("Year")[tagliche_bedarf.columns[2]].sum().reset_index()
    jahr_bedarf.columns = ["Jahr", "Strombedarf"]

    tag_bedarf = tagliche_bedarf.iloc[:, [0, 2]].copy()
    tag_bedarf.columns = ["Tag", "Strombedarf"]
    tag_bedarf.to_csv(os.path.join(consumption_path, "stromverbrauch_smard.csv"), index=False)

    tag_speicher = tagliche_bedarf.iloc[:, [0, 4]].copy()
    tag_speicher.columns = ["Tag", "Pumpspeicher"]
    tag_speicher.to_csv(os.path.join(storage_path, "pumpspeicher_smard.csv"), index=False)

    tag_residuallast = tagliche_bedarf.iloc[:, [0, 5]].copy()
    tag_residuallast.columns = ["Tag", "Residuallast"]
    tag_residuallast.to_csv(os.path.join(generation_path, "residuallast_smard.csv"), index=False)

    return jahr_bedarf, tag_bedarf


def iea_stromlebenstil(destatis_path, iea_path, country):
    """
    The following function generates a yearly electricity demand dataframe by multiplying the population of Germany
    obtained by destatis (Statistisches Bundesamt) with the electricity consumption per capita csv file obtained by
    the Internacional Energy Agency (iea). The data can represent the stromlebenstil data of multple countries,
    which is why the country value exists. The iea data is obtained from the following link:

    https://www.iea.org/countries/germany/electricity
    - Per-capita electricity consumption

    :param
        folder_path: Has to match the folder_path where the destatis population data is saved.

    :param
        iea_path: Has to match the iea_path where the destatis population data is saved.

    :param
        country: determines the country iea csv file to be used

    :return:
        jahr_bedarf: Electricity consumption dataframe (calculated from stromlebenstil)
    """
    bevolkerung = pd.read_csv(os.path.join(destatis_path, "destatis_germany_population.csv"))
    strom_lebenstil = pd.read_csv(os.path.join(iea_path, f"International Energy Agency - Electricity consumption per "
                                                         f"capita, {country}.csv"))

    merged = pd.merge(bevolkerung, strom_lebenstil, on="Year")
    jahr_bedarf = pd.DataFrame()
    jahr_bedarf["Jahr"] = merged["Year"]
    jahr_bedarf["Strombedarf"] = merged["Population"] * merged.iloc[:, 2]

    return jahr_bedarf


def iea_stromsektor(folder_path, iea_path, yearly_demand, country):
    """
    The following function generates a csv file that represents the percentage share of each sector found within the
    International Energy Agency (IEA) csv files, however due to the lack of information found within the disaggregated
    sector data, the total electricity consumption used that represents the 100% yearly demand come from the
    jahr_bedarf output value of past functions like smard_strombedarf and iea_stromlebenstil.

    The IEA sectoral data is obtained from:
    Evolution of electricity final consumption by sector since 2000
    https://www.iea.org/countries/germany/electricity


    :param
        iea_path: Has to match the iea_path where the destatis population data is saved.

    :param
        yearly_demand: Electricity consumption dataframe

    :param
        country: determines the country iea csv file to be used

    """
    sektor_dic = {
        'Agriculture / Forestry': 'Landwirtschaft',
        'Commercial and Public Services': 'Wirtschaft',
        'Industry': 'Industrie',
        'Residential': 'Haushalt',
        'Transport': 'Verkehr',
        'Fishing': 'Fischerei',
        'Other non - specified': 'Sonstiges'
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


def jahr_zu_tag(jahr_ziel, jahr_quelle, tag_quelle, folder_path):
    """
    Projects annual electricity demand data onto a daily resolution by scaling historical daily data.

    This function merges annual target data (`jahr_ziel`) with historical annual data (`jahr_quelle`)
    to compute yearly scaling ratios. These ratios are applied to the historical daily demand data
    (`tag_quelle`) to create a projected daily demand dataset (`tag_ziel`). The resulting dataset is
    saved as a CSV file in the specified folder and visualized using a comparison plot.

    :param
        jahr_ziel:  DataFrame containing generated annual electricity demand

    :param
        jahr_quelle: Original DataFrame containing historical annual electricity demand.

    :param
        tag_quelle: Daily dataframe that will determine the temporal components

    :param
        folder_path: Path to the directory where the generated daily CSV file will be saved.

    """
    jahr_data = pd.merge(jahr_ziel, jahr_quelle, on='Jahr', suffixes=('_ziel', '_quelle'))

    jahr_data['Ratio'] = jahr_data['Strombedarf_ziel'] / jahr_data['Strombedarf_quelle']
    print(jahr_data)
    ratio_mapping = jahr_data.set_index('Jahr')['Ratio'].to_dict()

    temp = tag_quelle.copy()
    temp['Jahr'] = pd.to_datetime(temp['Tag']).dt.year

    temp['Strombedarf'] = temp.apply(lambda row: round(row.iloc[1] * ratio_mapping.get(row['Jahr'], 1), 2), axis=1)
    tag_ziel = temp[['Tag', 'Strombedarf']]
    tag_ziel.to_csv(os.path.join(folder_path, "stromverbrauch_lebenstil.csv"), index=False)
    # Unique Vergleich graph
    oemof_graphs.vergleich_graphik(tag_ziel, tag_quelle, jahr_ziel, jahr_quelle)


smard_path = select_folder(message="SMARD")
smard_15min_csv_maker(smard_path, key_name="Stromverbrauch")
smard_15min_csv_maker(smard_path, key_name="Erzeugung")
consumption_path = select_folder(message="prep_consumption")
generation_path = select_folder(message="prep_generation")
storage_path = select_folder(message="prep_storage")

strombedarf_smard_jahr, strombedarf_smard_tag = smard_strombedarf(smard_path, consumption_path,
                                                                  storage_path, generation_path)

# Possible countries = ["Germany", "Denmark", "Norway", "Finland", "United States", "Sweden", "France"]
destatis_path = select_folder(message="DESTATIS")
iea_path = select_folder(message="IEA")

strombedarf_lebenstil_jahr = iea_stromlebenstil(destatis_path, iea_path, country="Germany")
iea_stromsektor(consumption_path, iea_path, strombedarf_smard_jahr, country="Germany")
# iea_stromsektor(folder_path, iea_path, strombedarf_lebenstil_jahr, country="Germany")
jahr_zu_tag(strombedarf_lebenstil_jahr, strombedarf_smard_jahr, strombedarf_smard_tag, consumption_path)
