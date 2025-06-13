import pandas as pd
import matplotlib.pyplot as plt

countries = {
    'denmark': 'Dänemark',
    'germany': 'Deutschland',
    'finland': 'Finnland',
    'norway': 'Norwegen',
    'portugal': 'Portugal',
    'sweden': 'Schweden',
    'france': 'Frankreich'
}

country_colors = {
    'denmark': '#c60c30',  # Red
    'germany': '#000000',  # Black
    'finland': '#003580',  # Blue
    'norway': '#ba0c2f',  # Dark red
    'portugal': '#006600',  # Green
    'sweden': '#FECC02',  # Swedish yellow
    'france': '#8806ce'  # French blue
}


def plot_energy_supply():
    plt.figure(figsize=(12, 6))

    for country, label in countries.items():
        file_path = f'data_understanding/iea/iea_{country}_electricity_consumption_capita.csv'

        try:
            df = pd.read_csv(file_path)
            years = df.iloc[:, 0]
            energy_per_capita = df.iloc[:, 1]

            plt.plot(years, energy_per_capita, label=label, color=country_colors[country])

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.title('Vergleich des Stromverbrauch pro Kopf in den IEA-Ländern')
    plt.xlabel('Jahr')
    plt.ylabel('MWh pro Kopf')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_energy_supply()

"""
data_paths = {}
for country in countries.keys():
    data_paths[country] = {
        'total_energy': f"iea/iea_{country}_total_energysupply.csv",
        'total_consumption': f"iea/iea_{country}_energy_consumption_sector.csv",
        'electricity_generation': f"iea/iea_{country}_electricity_generation.csv",
        'domestic_production': f"iea/iea_{country}_domestic_production.csv"
    }


def total_energy_supply(file_path, country):
    plt.figure(figsize=(10, 6))
    csv_file = pd.read_csv(file_path)
    coal_data = csv_file[csv_file.iloc[:, 0] == 'Coal']
    oil_data = csv_file[csv_file.iloc[:, 0] == 'Oil']
    gas_data = csv_file[csv_file.iloc[:, 0] == 'Natural gas']
    hydro_data = csv_file[csv_file.iloc[:, 0] == 'Hydro']
    wind_data = csv_file[csv_file.iloc[:, 0] == 'Wind, sonne, etc.']
    bio_data = csv_file[csv_file.iloc[:, 0] == 'Biofuels and waste']

    plt.plot(coal_data['Year'], coal_data['Value'], marker='o', linestyle='-', label='Kohle', color="orange")
    plt.plot(oil_data['Year'], oil_data['Value'], marker='o', linestyle='-', label='Öl', color="black")
    plt.plot(gas_data['Year'], gas_data['Value'], marker='o', linestyle='-', label='Erdgas', color="red")
    plt.plot(hydro_data['Year'], hydro_data['Value'], marker='o', linestyle='-', label='Wasserkraft', color="blue")
    plt.plot(wind_data['Year'], wind_data['Value'], marker='o', linestyle='-', label='Wind- und Solarenergie',
             color="purple")
    plt.plot(bio_data['Year'], bio_data['Value'], marker='o', linestyle='-', label='Biokraftstoffe und Bioabfall',
             color="green")
    min_year = csv_file['Year'].min()
    max_year = csv_file['Year'].max()
    plt.title(f'Gesamtenergieversorgung in {country} von {min_year} bis {max_year}')
    plt.xlabel('Jahr')
    plt.ylabel('Energieversorgung (TJ)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.07, bottom=0.078, right=0.827, top=0.943)
    plt.grid(True)
    plt.show()


def total_consumption_sector(file_path, country):
    plt.figure(figsize=(10, 6))
    csv_file = pd.read_csv(file_path)
    industry_data = csv_file[csv_file.iloc[:, 0] == 'Industry']
    transport_data = csv_file[csv_file.iloc[:, 0] == 'Transport']
    residential_data = csv_file[csv_file.iloc[:, 0] == 'Residential']
    commercial_data = csv_file[csv_file.iloc[:, 0] == 'Commercial and public services']
    agriculture_data = csv_file[csv_file.iloc[:, 0] == 'Agriculture / forestry']

    plt.plot(industry_data['Year'], industry_data['Value'], marker='o', linestyle='-', label='Industrie',
             color="orange")
    plt.plot(transport_data['Year'], transport_data['Value'], marker='o', linestyle='-', label='Transport',
             color="black")
    plt.plot(residential_data['Year'], residential_data['Value'], marker='o', linestyle='-', label='Privater',
             color="red")
    plt.plot(commercial_data['Year'], commercial_data['Value'], marker='o', linestyle='-', label='Wirtschaftliche',
             color="blue")
    plt.plot(agriculture_data['Year'], agriculture_data['Value'], marker='o', linestyle='-', label='Landwirtschaft',
             color="green")

    min_year = csv_file['Year'].min()
    max_year = csv_file['Year'].max()
    plt.title(f'Gesamtendenergieverbrauch pro Sektor in {country} von {min_year} bis {max_year}')
    plt.xlabel('Jahr')
    plt.ylabel('Energieverbrauch (TJ)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.07, bottom=0.078, right=0.886, top=0.943)
    plt.grid(True)
    plt.show()


def total_electricity_generation(file_path, country):
    plt.figure(figsize=(10, 6))
    csv_file = pd.read_csv(file_path)
    coal_data = csv_file[csv_file.iloc[:, 0] == 'Coal']
    oil_data = csv_file[csv_file.iloc[:, 0] == 'Oil']
    gas_data = csv_file[csv_file.iloc[:, 0] == 'Natural gas']
    hydro_data = csv_file[csv_file.iloc[:, 0] == 'Hydro']
    bio_data = csv_file[csv_file.iloc[:, 0] == 'Biofuels']
    waste_data = csv_file[csv_file.iloc[:, 0] == 'Waste']
    wind_data = csv_file[csv_file.iloc[:, 0] == 'Wind']
    solar_data = csv_file[csv_file.iloc[:, 0] == 'Solar PV']

    plt.plot(coal_data['Year'], coal_data['Value'], marker='o', linestyle='-', label='Kohle', color="orange")
    plt.plot(oil_data['Year'], oil_data['Value'], marker='o', linestyle='-', label='Öl', color="black")
    plt.plot(gas_data['Year'], gas_data['Value'], marker='o', linestyle='-', label='Erdgas', color="red")
    plt.plot(hydro_data['Year'], hydro_data['Value'], marker='o', linestyle='-', label='Wasserkraft', color="blue")
    plt.plot(bio_data['Year'], bio_data['Value'], marker='o', linestyle='-', label='Biokraftstoffe',
             color="green")
    plt.plot(waste_data["Year"], waste_data["Value"], marker="o", linestyle="-", label="Abfall")
    plt.plot(wind_data['Year'], wind_data['Value'], marker='o', linestyle='-', label='Windenergie',
             color="cyan")
    plt.plot(solar_data['Year'], solar_data['Value'], marker='o', linestyle='-', label='PV-Solarenergie',
             color="yellow")
    min_year = csv_file['Year'].min()
    max_year = csv_file['Year'].max()
    plt.title(f'Gesamtstromerzeugung in {country} von {min_year} bis {max_year}')
    plt.xlabel('Jahr')
    plt.ylabel('Stromproduktion (GWh)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.07, bottom=0.078, right=0.878, top=0.943)
    plt.grid(True)
    plt.show()


def total_domestic_production(file_path, country):
    plt.figure(figsize=(10, 6))
    csv_file = pd.read_csv(file_path)
    oil_data = csv_file[csv_file.iloc[:, 0] == 'Crude oil']
    gas_data = csv_file[csv_file.iloc[:, 0] == 'Natural gas']
    hydro_data = csv_file[csv_file.iloc[:, 0] == 'Hydro']
    heat_data = csv_file[csv_file.iloc[:, 0] == 'Heat']
    wind_data = csv_file[csv_file.iloc[:, 0] == 'Wind, sonne, etc.']
    bio_data = csv_file[csv_file.iloc[:, 0] == 'Biofuels and waste']
    nuclear_data = csv_file[csv_file.iloc[:, 0] == 'Nuclear']

    plt.plot(oil_data['Year'], oil_data['Value'], marker='o', linestyle='-', label='Erdöl', color="black")
    plt.plot(gas_data['Year'], gas_data['Value'], marker='o', linestyle='-', label='Erdgas', color="red")
    plt.plot(hydro_data['Year'], hydro_data['Value'], marker='o', linestyle='-', label='Wasserkraft', color="blue")
    plt.plot(wind_data['Year'], wind_data['Value'], marker='o', linestyle='-', label='Wind- und Solarenergie',
             color="purple")
    plt.plot(heat_data['Year'], heat_data['Value'], marker='o', linestyle='-', label='Wärmeenergie', color="orange")
    plt.plot(bio_data['Year'], bio_data['Value'], marker='o', linestyle='-', label='Biokraftstoffe und Bioabfall',
             color="green")
    plt.plot(nuclear_data['Year'], nuclear_data['Value'], marker='o', linestyle='-', label='Atomkraft', color="yellow")
    min_year = csv_file['Year'].min()
    max_year = csv_file['Year'].max()
    plt.title(f'Nationale energieerzeugung in {country} von {min_year} bis {max_year}')
    plt.xlabel('Jahr')
    plt.ylabel('Energieerzeugung (TJ)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.07, bottom=0.078, right=0.827, top=0.943)
    plt.grid(True)
    plt.show()


def net_imports_exports(file_path, country="germany"):
    df = pd.read_csv(file_path)

    df_imports = df[df.iloc[:, 0] == 'Imports']
    df_exports = df[df.iloc[:, 0] == 'Exports']
    imports_by_year = df_imports.groupby('Year')['Value'].sum()
    exports_by_year = df_exports.groupby('Year')['Value'].sum()
    net_by_year = imports_by_year.add(exports_by_year, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_imports['Year'],
        df_imports['Value'],
        label='Importe',
        marker='o',
        color="green"
    )

    ax.plot(
        df_exports['Year'],
        df_exports['Value'],
        label='Exporte',
        marker='o',
        color="orange"
    )

    ax.plot(
        net_by_year.index,
        net_by_year.values,
        label='Netto-Strom',
        marker='o',
        color='black'
    )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"Stromimporte und -exporte in {country}")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Energie (TJ)")
    ax.legend()

    plt.tight_layout()
    plt.show()


net_imports_exports(f"iea/iea_germany_imports_exports.csv", country="Deutschland")

for country_code, country_name in countries.items():
    total_energy = data_paths[country_code]['total_energy']
    total_energy_supply(total_energy, country=country_name)

for country_code, country_name in countries.items():
    total_consumption = data_paths[country_code]['total_consumption']
    total_consumption_sector(total_consumption, country=country_name)

for country_code, country_name in countries.items():
    total_generation = data_paths[country_code]['electricity_generation']
    total_electricity_generation(total_generation, country=country_name)

for country_code, country_name in countries.items():
    total_domestic = data_paths[country_code]['domestic_production']
    total_domestic_production(total_domestic, country=country_name)

country_list = ["DNK", "DEU", "FIN", "NOR", "PRT", "SWE", "FRA"]
european_uno = "iea/iea_european_electricity_consumption.csv"
european_dos = "iea/iea_european_energy_supply_per_unit_gdp.csv"
european_tres = "iea/iea_european_renewables_share.csv"


file_path = '/Users/ochoa/PycharmProjects/Masterarbeit/smard/tag_Stromverbrauch.csv'
df = pd.read_csv(file_path, delimiter=';')
for col in df.columns[2:]:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%d.%m.%Y')
df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
print(df.iloc[:, 2].head())
df.dropna(inplace=True)

df.set_index(df.iloc[:, 0], inplace=True)

plt.figure(figsize=(15, 5))
plt.plot(df.index, df.iloc[:, 2], label='Tägliche Netzlast (MWh)', color="#0033CC")
plt.title('Historische tägliche Netzlast in Deutschland (SMARD.de)')
plt.xlabel('Datum')
plt.ylabel('Netzlast (MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""