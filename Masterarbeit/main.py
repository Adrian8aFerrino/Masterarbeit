import os
import pandas as pd
from oemof.tools import economics
from oemof.solph import EnergySystem, Bus, Flow, Investment
from oemof.solph.components import Sink, Converter, Source
from prep_consumption.oemof_consumption import sektor_demand_chow_lin
from prep_transmission.oemof_transmission import monte_carlo_loss
from prep_generation.oemof_generation import lcoe_generation, ep_costs_generation, offset_generation
from config import select_folder

country = "Germany"
consumption_path = select_folder(message="Consumption")
generation_path = select_folder(message="Generation")
worldbank_path = select_folder(message="Worldbank")

loss_df = pd.read_csv(os.path.join(worldbank_path, f"worldbank_loss_{country}.csv"), delimiter=",")
tag_bedarf = pd.read_csv(os.path.join(consumption_path, "stromverbrauch_smard.csv"), delimiter=",")
sektor_df = pd.read_csv(os.path.join(consumption_path, "stromsektor_prozent_{country}.csv"), delimiter=",")

tag_bedarf["Tag"] = pd.to_datetime(tag_bedarf["Tag"])
datetime_index = pd.date_range(start=tag_bedarf["Tag"].min(), end=tag_bedarf["Tag"].max(), freq='D')

# Main oemof model structure
energy_system = EnergySystem(timeindex=datetime_index)
strom_bedarf = Bus(label="electricity_demand")
plus_bedarf = Bus(label="electricity_demand_with_loss")
energy_system.add(strom_bedarf, plus_bedarf)

# Sinks (Sektoren: Fischerei, Haushalt, Industrie, Landwirtschaft, Verkehr, Wirtschaft)
sektor_name = "Industrie"
industrie_sink = Sink(
    label=f"{sektor_name}_bedarf",
    inputs={strom_bedarf: Flow(
        fix=sektor_demand_chow_lin(tag_bedarf, sektor_df, sektor_name),
        nominal_value=1
    )})
energy_system.add(industrie_sink)

sektor_name = "Haushalt"
haushalt_sink = Sink(
    label=f"{sektor_name}_bedarf",
    inputs={strom_bedarf: Flow(
        fix=sektor_demand_chow_lin(tag_bedarf, sektor_df, sektor_name),
        nominal_value=1
    )})
energy_system.add(haushalt_sink)

sektor_name = "Verkehr"
verkehr_sink = Sink(
    label=f"{sektor_name}_bedarf",
    inputs={strom_bedarf: Flow(
        fix=sektor_demand_chow_lin(tag_bedarf, sektor_df, sektor_name),
        nominal_value=1
    )})
energy_system.add(verkehr_sink)

sektor_name = "Wirtschaft"
wirtschaft_sink = Sink(
    label=f"{sektor_name}_bedarf",
    inputs={strom_bedarf: Flow(
        fix=sektor_demand_chow_lin(tag_bedarf, sektor_df, sektor_name),
        nominal_value=1
    )})
energy_system.add(wirtschaft_sink)

# Transmission and distribution losses
transmission_losses = Converter(
    label=f"transmission_{sektor_name}_losses",
    inputs={plus_bedarf: Flow()},
    outputs={strom_bedarf: Flow()},
    conversion_factors={strom_bedarf: monte_carlo_loss(worldbank_path, country)}
)

energy_system.add(transmission_losses)

# GENERATION UNITS
# ['Sonne' 'Onshore-Wind' 'Offshore-Wind' 'Erdgas' 'Biomasse' 'Kernkraft' 'Kohle']

kapazitat = pd.read_csv(os.path.join(generation_path, "kapazitat_technologie.csv"), delimiter=",")

sonne = kapazitat[kapazitat["Technologie"] == "Sonne"]
offshore = kapazitat[kapazitat["Technologie"] == "Offshore-Wind"]
onshore = kapazitat[kapazitat["Technologie"] == "Onshore-Wind"]
erdgas = kapazitat[kapazitat["Technologie"] == "Erdgas"]
kernkraft = kapazitat[kapazitat["Technologie"] == "Kernkraft"]
kohle = kapazitat[kapazitat["Technologie"] == "Kohle"]

# Sonne
for c_solar in sonne["Cluster"].unique():
    cluster_df = sonne[sonne["Cluster"] == c_solar]
    max_solar = cluster_df["Nettonennleistung (elektrische Wirkleistung) in MW"].quantile(0.75)

    solar_source = Source(
        label=f"solar_cluster_{int(c_solar)}",
        outputs={
            plus_bedarf: Flow(
                investment=Investment(
                    ep_costs=ep_costs_generation(tech="Sonne"),
                    minimum=0,
                    maximum=max_solar,
                    nonconvex=True,
                    offset=offset_generation("Sonne")
                ),
                variable_costs=lcoe_generation(tech="Sonne")
            )
        }
    )
    energy_system.add(solar_source)

# Onshore
for c_onshore in sonne["Cluster"].unique():
    cluster_df = sonne[sonne["Cluster"] == c_solar]
    max_solar = cluster_df["Nettonennleistung (elektrische Wirkleistung) in MW"].quantile(0.75)

    solar_source = Source(
        label=f"solar_cluster_{int(c_solar)}",
        outputs={
            plus_bedarf: Flow(
                investment=Investment(
                    ep_costs=ep_costs_generation(),
                    minimum=0,
                    maximum=max_solar,
                    nonconvex=True,
                    offset=offset_generation()
                ),
                variable_costs=lcoe_generation(tech="Sonne")
            )
        }
    )
    energy_system.add(solar_source)