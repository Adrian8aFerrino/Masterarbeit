import pandas as pd
from oemof.tools import economics


# Open Power System Data (OPSD): Open-source platform for electricity system modeling dedicated on data collection:
# https://open-power-system-data.org/

def capex_generation(tech):  # CAPEX = Capital Expenditure (serves calculate annuity costs for each source)
    return


def wacc_generation(tech):  # WACC = Weighted Average Costs of Capital
    return


def lcoe_generation(tech):  # LCOE = Levelized Cost of Electricity (LCOE) (serves calculating the generation costs of electricity
    return


def ep_costs_generation(tech):
    if tech == "Sonne":
        tech_generation = 0
    return


def offset_generation(tech):
    if tech == "Sonne":
        offset_generation = 0

    return


capex = 1000
wacc = 0.05
ep_costs_solar = economics.annuity(capex=capex, n=lifetime, wacc=wacc)
lcoe = 60  # €/MWh
offset = 250000
