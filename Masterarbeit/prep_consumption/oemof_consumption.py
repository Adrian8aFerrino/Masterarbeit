import pandas as pd


def sektor_demand_chow_lin(tag_bedarf, sektor_df, sektor_name):
    if sektor_name not in sektor_df.columns:
        print(f"Error: Sector name '{sektor_name}' not found in sektor_df columns.")
        return None, None
    tag_bedarf['Tag'] = pd.to_datetime(tag_bedarf['Tag'], errors='coerce')
    tag_bedarf["Jahr"] = tag_bedarf["Tag"].dt.year
    sektor_df = sektor_df[["Jahr", sektor_name]]
    sektor_df = sektor_df.dropna()

    merge_df = pd.merge(tag_bedarf, sektor_df, on="Jahr", how="inner")
    jahr_df = merge_df.groupby("Jahr")["Strombedarf"].sum().reset_index(name="Jahresbedarf")
    merge_df = pd.merge(merge_df, jahr_df, on="Jahr")
    merge_df["sektor_ratio"] = merge_df["Strombedarf"] / merge_df["Jahresbedarf"]  # daily share of annual demand
    merge_df["Sektor_bedarf_jahr"] = merge_df[sektor_name] * merge_df["Jahresbedarf"] / 100.0  # annual sector demand
    merge_df["Sektor_bedarf_tag"] = (merge_df["sektor_ratio"] * merge_df["Sektor_bedarf_jahr"]).round(2)

    result_df = merge_df[['Tag', 'Sektor_bedarf_tag']].rename(
        columns={'Sektor_bedarf_tag': 'Strombedarf'})
    result_df['Tag'] = pd.to_datetime(result_df['Tag'])
    result_df.set_index('Tag', inplace=True)

