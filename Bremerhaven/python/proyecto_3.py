# Part 1: Import libraries and data
import numpy as np
import pandas as pd
import seaborn as sns
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib.animation import FuncAnimation


pd.set_option('display.max_columns', None)

df_agent_flow = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/agent_flow.csv",
                            delimiter=",", header=0)
df_raw_material = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/raw_material.csv",
                              delimiter=",", header=0)
df_reactor_sensor = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/reactor_sensor.csv",
                                delimiter=",", header=0)
df_dri_sample = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/dri_sample.csv",
                            delimiter=",", header=0)

# Part 2: Data preparation for RAW MATERIAL, REACTOR SENSOR, AGENT FLOW and DRI SAMPLES
try:
    df_raw_material['Date'] = pd.to_datetime(df_raw_material['Date'], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_raw_material['Date'] = pd.to_datetime(df_raw_material['Date'], format='mixed')
df_raw_material = df_raw_material.sort_values(by='Date')
df_raw_material['Date'] = df_raw_material['Date'] - pd.Timedelta(hours=9)
df_raw_material.set_index('Date', inplace=True)

try:
    df_reactor_sensor['Date'] = pd.to_datetime(df_reactor_sensor['Date'], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_reactor_sensor['Date'] = pd.to_datetime(df_reactor_sensor['Date'], format='mixed')

df_reactor_sensor = df_reactor_sensor.sort_values(by='Date')
df_reactor_sensor.set_index('Date', inplace=True)
df_reactor_sensor = df_reactor_sensor.resample('30T').agg(
    {'Temperature 1': ['mean', 'std'], 'Pressure 1': ['mean', 'std'],
     'Temperature 2': ['mean', 'std'], 'Pressure 2': ['mean', 'std'],
     'Temperature 3': ['mean', 'std'], 'Pressure 3': ['mean', 'std'],
     'Temperature 4': ['mean', 'std'], 'Pressure 4': ['mean', 'std'],
     'Temperature 5': ['mean', 'std'], 'Pressure 5': ['mean', 'std']})
df_reactor_sensor = df_reactor_sensor.rolling("9H").mean()
df_reactor_sensor = df_reactor_sensor.iloc[18:]

try:
    df_agent_flow['Date'] = pd.to_datetime(df_agent_flow["Date"], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_agent_flow['Date'] = pd.to_datetime(df_agent_flow['Date'], format='mixed')

df_agent_flow = df_agent_flow.sort_values(by='Date')
df_agent_flow.set_index('Date', inplace=True)
df_agent_flow = df_agent_flow.resample('30T').agg({'H2': ['mean', 'std'], 'CO': ['mean', 'std'],
                                                   'CH4': ['mean', 'std'], 'Al2O3': ['mean', 'std'],
                                                   '4SiO2': ['mean', 'std'], 'C6H10O5': ['mean', 'std'],
                                                   'C12H22O11': ['mean', 'std'], 'CaCO3': ['mean', 'std'],
                                                   'CaMgCO3': ['mean', 'std'], '3SiO2': ['mean', 'std']})
df_agent_flow = df_agent_flow.rolling("9H").mean()
df_agent_flow = df_agent_flow.iloc[18:]

try:
    df_dri_sample['Date'] = pd.to_datetime(df_dri_sample["Date"], format='%d/%m/%Y %H:%M')
except ValueError as e:
    df_dri_sample['Date'] = pd.to_datetime(df_dri_sample['Date'], format='mixed')

df_dri_sample = df_dri_sample.sort_values(by='Date')
df_dri_sample.columns = [df_dri_sample.columns[0]] + [col + "_dri" for col in df_dri_sample.columns[1:]]

final_variables = pd.concat([df_raw_material, df_reactor_sensor, df_agent_flow], axis=1,
                            join="outer").sort_index().ffill()
final_variables = final_variables.fillna(final_variables.mean())
complete_df = pd.merge(final_variables, df_dri_sample, on='Date')
outcome_variables = np.array(df_dri_sample.drop(columns=["Date"]))
print("Independent_variables", final_variables.shape)
print("Dependent_variables", df_dri_sample.shape)
print("Complete_variables", complete_df.shape)

# Part 3: Correlation matrices (PEARSON & SPEARMAN)
pearson_corr = final_variables.corr(method="pearson")
spearman_corr = final_variables.corr(method="spearman")

with pd.ExcelWriter('databases/correlation_matrices.xlsx') as writer:
    pearson_corr.to_excel(writer, sheet_name='Pearson Correlation')
    spearman_corr.to_excel(writer, sheet_name='Spearman Correlation')

fig1, ax1 = plt.subplots(1, 2)
df_pearson = pd.read_excel('/Users/ochoa/PycharmProjects/Bremerhaven/databases/correlation_matrices.xlsx',
                           sheet_name='Pearson Correlation', header=0, index_col=0)
sns.heatmap(data=df_pearson, annot=False, cmap="Blues", ax=ax1[0], xticklabels=False, yticklabels=False)
ax1[0].set_title("Pearson correlation (DRI process)")

df_spearman = pd.read_excel('/Users/ochoa/PycharmProjects/Bremerhaven/databases/correlation_matrices.xlsx',
                            sheet_name='Spearman Correlation', header=0, index_col=0)
sns.heatmap(data=df_spearman, annot=False, cmap="Greens", ax=ax1[1], xticklabels=False, yticklabels=False)
ax1[1].set_title("Spearman correlation (DRI process)")
plt.show()


# Part 4: MinMax Scaling and PCA
scaler = MinMaxScaler()
independent_variables = complete_df.iloc[:, 1:51].copy()
independent_variables.columns = independent_variables.columns.astype(str)
independent_variables = pd.DataFrame(scaler.fit_transform(independent_variables), columns=independent_variables.columns)
print("ALL columns", independent_variables.columns)
pca_dri = PCA()
pca_dri.fit(independent_variables)
principal_components = pca_dri.transform(independent_variables)
selected_components = principal_components[:, :6]

# Part 5: Animation Cumulative Explained Variance
fig, ax = plt.subplots()
line, = ax.plot([], [], color='orange')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.5)
ax.set_title('Cumulative Explained Variance vs Number of Components')


def update(frame):
    components = frame + 1
    cumulative_variance = np.cumsum(pca_dri.explained_variance_ratio_)[:components]
    line.set_data(np.arange(1, components + 1), cumulative_variance)
    return line,


ani = FuncAnimation(fig, update, frames=len(pca_dri.explained_variance_ratio_), interval=200)
plt.show()


# Part 6: Generalized additive model (GAM) regression model
gams = []
dict_of_lists = {"DRI_pellet_dri": ['Abrasion', 'C', 'CaO', 'Fe2O3', 'K2Cr2O7', 'MgO'],
                 "Pellet_size_dri": ['CaO', 'K2Cr2O7', 'Ore_size', "('C12H22O11', 'std')", "('CaMgCO3', 'std')",
                                     "('Pressure 5', 'mean')"],
                 "Fe_dri": ['Abrasion', 'Al2O3', 'C', 'Fe2O3', 'K2Cr2O7', 'MgO'],
                 "S_dri": ['Fe2O3', 'S', "('Pressure 5', 'mean')", "('Pressure 5', 'std')",
                           "('Temperature 5', 'mean')", "('Temperature 5', 'std')"],
                 "C_dri": ['Abrasion', 'Al2O3', 'C', 'CaO', 'K2Cr2O7', 'S'],
                 "Metallization_dri": ['Fe2O3', 'K2Cr2O7', 'MgO', 'Porosity', "('CaCO3', 'mean')", "('H2', 'mean')"],
                 "Slag_dri": ["('3SiO2', 'mean')", "('CaMgCO3', 'mean')", "('Pressure 1', 'std')",
                              "('Pressure 2', 'mean')", "('Pressure 5', 'mean')", "('Pressure 5', 'std')"],
                 "Waste_powder_dri": ['CaO', 'Fe2O3', 'Ore_size', 'Porosity', "('C12H22O11', 'std')",
                                      "('CaMgCO3', 'std')"],
                 "Porosity_dri": ['C', 'Fe2O3', 'K2Cr2O7', 'MgO', 'Porosity', "('C12H22O11', 'std')"]}
filtered_arrays = {key: independent_variables[columns].to_numpy() for key, columns in dict_of_lists.items()}
filtered_arrays = list(filtered_arrays.items())

print("PCA shape", selected_components.shape)
print("PCA type", type(selected_components))
for i in range(outcome_variables.shape[1]):
    key, value = filtered_arrays[i]
    gam_1 = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5)).fit(selected_components, outcome_variables[:, i])
    gam_2 = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5)).fit(value, outcome_variables[:, i])
    gams.append(gam_1)
    print(f"Summary for target variable {df_dri_sample.columns[i+1]}")
    print(gam_1.summary())
    print(gam_2.summary())
    XX = gam_1.generate_X_grid(term=0)
    plt.figure()
    plt.plot(XX[:, 0], gam_1.predict(XX), label='LinearGAM', color="black", linewidth=2.5)
    plt.plot(XX[:, 0], gam_2.predict(XX), label='Stepwise Regression', color="red", linewidth=2.5)
    plt.scatter(selected_components[:, 0], outcome_variables[:, i], color='orange', label='Expected values', alpha=0.25)
    plt.title(f'LinearGAM for target variable {df_dri_sample.columns[i+1]}')
    plt.xlabel('First Principal Component Value')
    plt.ylabel(f'Target Variable {df_dri_sample.columns[i+1]}')
    plt.legend()
    plt.show()
# predictions = np.column_stack([gam.predict(selected_components) for gam in gams])


# Part 7: Reverse Stepwise Regression
import pandas as pd
from statsmodels.formula.api import ols
import re

print(type(outcome_variables))
outcome_variables = pd.DataFrame(df_dri_sample, columns=df_dri_sample.columns[1:])


def sanitize_column_names(columns):
    return [re.sub(r'\W+', '_', col) for col in columns]


def backward_elimination(data, dependent_var, n_vars=6):
    independent_vars = data.columns.difference([dependent_var])
    independent_vars = [f"Q('{var}')" for var in independent_vars]
    while len(independent_vars) > n_vars:
        formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
        model = ols(formula, data).fit()
        p_values = model.pvalues.iloc[1:]
        max_p_value_var = p_values.idxmax()
        if p_values[max_p_value_var] > 0.05:
            independent_vars = [item for item in independent_vars if item != max_p_value_var]
        else:
            break
    return independent_vars


# Apply the backward elimination for each dependent variable
final_vars = {}
for dep_var in outcome_variables.columns:
    data = pd.concat([independent_variables, outcome_variables[dep_var]], axis=1)
    data.columns = sanitize_column_names(data.columns)
    final_vars[dep_var] = backward_elimination(data, dep_var)
    print(f"Columns for {dep_var}", final_vars[dep_var])
