import os
import pandas as pd
import numpy as np


def monte_carlo_loss(worldbank_path, country):
    loss_df = pd.read_csv(os.path.join(worldbank_path, f"worldbank_loss_{country}.csv"), delimiter=",")
    np.random.seed(1996)

    loss_df['Jahr'] = pd.to_datetime(loss_df['Jahr'], format='%Y')
    loss_df['weight'] = loss_df['Jahr'].rank().astype(int)
    loss_df['Verluste %'] = loss_df['Verluste %'] / 100
    loss_fraction = loss_df['Verluste %'].values
    weight = loss_df['weight'].values
    weighted_mean = np.average(loss_fraction, weights=weight)

    def weighted_std(loss_fraction, weight):
        average = np.average(loss_fraction, weights=weight)
        variance = np.average((loss_fraction - average)**2, weights=weight)
        return np.sqrt(variance)

    weighted_standard_dev = weighted_std(loss_fraction, weight)
    simulated_losses = np.random.normal(loc=weighted_mean, scale=weighted_standard_dev, size=300)
    simulated_losses = np.clip(simulated_losses, 0.02, 0.06)
    conversion_factor = 1 - simulated_losses
    return conversion_factor[0]
