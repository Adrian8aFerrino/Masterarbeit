# Part 1: Libraries and data download
import time
import uuid
import warnings
import datetime
import pulp as plp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis, jarque_bera
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

staff_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/p_zwei_staff_elements.csv", encoding='latin1',
                       header=0)
store_df = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/p_zwei_store_elements.csv", encoding='latin1',
                       header=0, dtype={'Beleggröße': str, 'Produktverkäufe': str})


def expand_staff_df_randomly(staff_df, copies_per_row=10):
    # Clean strings and define value options from original data
    schicht_opts = staff_df['Schichtverfügbarkeit'].dropna().unique().tolist()
    quali_opts = staff_df['Qualifikationen'].dropna().unique().tolist()
    overtime_opts = staff_df['Überstunden'].dropna().unique().tolist()
    pref_opts = staff_df['Schichtpräferenz'].dropna().unique().tolist()
    store_opts = staff_df['Gemischtwarenladen'].dropna().unique().tolist()

    # Dummy name lists
    first_names = ["Alex", "Jamie", "Taylor", "Jordan", "Morgan", "Casey", "Avery", "Quinn"]
    last_names = ["Smith", "Johnson", "Lee", "Brown", "Garcia", "Martinez", "Davis", "Miller"]

    new_rows = []

    for _, row in staff_df.iterrows():
        for _ in range(copies_per_row):
            new_rows.append({
                'ID Nummer': 'ID' + str(uuid.uuid4())[:8],
                'Vorname': np.random.choice(first_names),
                'Nachname': np.random.choice(last_names),
                'Schichtverfügbarkeit': np.random.choice(schicht_opts),
                'Qualifikationen': np.random.choice(quali_opts),
                'Überstunden': np.random.choice(overtime_opts),
                'Schichtpräferenz': np.random.choice(pref_opts),
                'Einstellungsdatum': row['Einstellungsdatum'],
                'Entlassungsdatum': row['Entlassungsdatum'],
                'Gemischtwarenladen': np.random.choice(store_opts)
            })

    expanded_df = pd.DataFrame(new_rows)
    return expanded_df


expanded_staff_df = expand_staff_df_randomly(staff_df, copies_per_row=10)
staff_df = pd.concat([staff_df, expanded_staff_df], ignore_index=True)


# Part 2: Data preparation and dummy variables
staff_df = staff_df.replace('k.A', np.nan).fillna('31/03/2023')
dum_uno = staff_df['Schichtverfügbarkeit'].str.get_dummies()
dum_dos = staff_df['Qualifikationen'].str.get_dummies()
dum_tres = staff_df['Gemischtwarenladen'].astype(str).str.get_dummies()

staff_df[['Überstunden', 'Schichtpräferenz']] = staff_df[['Überstunden', 'Schichtpräferenz']]\
    .replace({'JA': 1, 'NEIN': 0}).astype(int)
staff_df['Einstellungsdatum'] = pd.to_datetime(staff_df['Einstellungsdatum'], format='%d/%m/%Y')
staff_df['Entlassungsdatum'] = pd.to_datetime(staff_df['Entlassungsdatum'], format='%d/%m/%Y')
staff_df['Tage_zwischen'] = (staff_df['Entlassungsdatum'] - staff_df['Einstellungsdatum']).dt.days
staff_model = staff_df.copy(deep=True)
staff_retention = staff_df.copy(deep=True)
store_df['Datum'] = pd.to_datetime(store_df['Datum'], dayfirst=True)
store_df['Tag_des_Jahres'] = store_df['Datum'].dt.dayofyear
store_df['Gemischtwarenladen'] = store_df['Gemischtwarenladen'].astype(str)
store_df[['Feiertag', 'Sonderaktionen']] = store_df[['Feiertag', 'Sonderaktionen']]\
    .replace({'JA': 1, 'NEIN': 0}).astype(int)

store_df['Beleggröße'] = store_df['Beleggröße'].str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '.').astype(float)
store_df['Produktverkäufe'] = (
    store_df['Produktverkäufe']
    .str.replace(r'[^\d,.-]', '', regex=True)  # remove €, spaces, etc.
    .str.replace('.', '', regex=False)        # remove thousands separator
    .str.replace(',', '.', regex=False)       # convert decimal comma to dot
    .astype(float)
)

store_model = store_df.copy(deep=True)
store_retention = store_df.copy(deep=True)


# Part 3: custom_describe() generates descriptive statistics for
def custom_describe(df):
    numeric_df = df.select_dtypes(include=[np.number])
    description = numeric_df.describe()
    description.loc['skewness'] = numeric_df.apply(skew)
    description.loc['kurtosis'] = numeric_df.apply(kurtosis)
    description.loc['jarque-bera'] = numeric_df.apply(lambda x: jarque_bera(x)[1])
    return description


# Part 4: Descriptive statistics (Measures of central tendency, measures of variability and Measures of frequency)
new_store_df = store_df[store_df['Produktverkäufe'] != 0]

gs_eins = gridspec.GridSpec(2, 2)
fig_eins = plt.figure()
ax1_eins = fig_eins.add_subplot(gs_eins[0, 0])
ax1_eins.plot([0, 1])
ax2_eins = fig_eins.add_subplot(gs_eins[0, 1])
ax2_eins.plot([0, 1])
ax3_eins = fig_eins.add_subplot(gs_eins[1, 0])
ax3_eins.plot([0, 1])
ax4_eins = fig_eins.add_subplot(gs_eins[1, 1])
ax4_eins.plot([1, 1])

fig_eins.suptitle('Häufigkeiten (Personal)')
staff_count1 = staff_df['Schichtverfügbarkeit'].value_counts()
staff_count1.plot(kind='bar', ax=ax1_eins, color='navy')
ax1_eins.tick_params(labelrotation=0)
ax1_eins.set_title('Schichtverfügbarkeit')
for rect in ax1_eins.patches:
    height = rect.get_height()
    ax1_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count2 = staff_df['Qualifikationen'].value_counts()
staff_count2.plot(kind='bar', ax=ax2_eins, color='darkred')
ax2_eins.tick_params(labelrotation=0)
ax2_eins.set_title('Qualifikationen')
for rect in ax2_eins.patches:
    height = rect.get_height()
    ax2_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count3 = staff_df['Gemischtwarenladen'].value_counts()
staff_count3.plot(kind='bar', ax=ax3_eins, color='orange')
ax3_eins.tick_params(labelrotation=0)
ax3_eins.set_title('Gemischtwarenladen')
for rect in ax3_eins.patches:
    height = rect.get_height()
    ax3_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

staff_count4 = staff_df['Schichtpräferenz'].value_counts()
staff_count4.plot(kind='bar', ax=ax4_eins, color='green')
ax4_eins.tick_params(labelrotation=0)
ax4_eins.set_title('Schichtpräferenz')
for rect in ax4_eins.patches:
    height = rect.get_height()
    ax4_eins.text(rect.get_x() + rect.get_width() / 2, height, int(height), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
# plt.show()

# Part 5: Descriptive statistics (Measures of distribution)
gs_zwei = gridspec.GridSpec(2, 2)
fig_zwei = plt.figure()
ax1_zwei = fig_zwei.add_subplot(gs_zwei[0, 0])
ax1_zwei.plot([0, 1])
ax2_zwei = fig_zwei.add_subplot(gs_zwei[0, 1])
ax2_zwei.plot([0, 1])
ax3_zwei = fig_zwei.add_subplot(gs_zwei[1, 0])
ax3_zwei.plot([0, 1])
ax4_zwei = fig_zwei.add_subplot(gs_zwei[1, 1])
ax4_zwei.plot([1, 1])

fig_zwei.suptitle('Maße der Häufigkeitsverteilung')
store_hist1 = new_store_df['Kundenfrequenz']
store_hist1.plot(kind='hist', ax=ax1_zwei, color='darkcyan', bins=30)
ax1_zwei.tick_params(labelrotation=0)
ax1_zwei.set_title('Kundenfrequenz')

store_hist2 = new_store_df['Beleggröße']
store_hist2.plot(kind='hist', ax=ax2_zwei, color='firebrick', bins=30)
ax2_zwei.tick_params(labelrotation=0)
ax2_zwei.set_title('Beleggröße')

store_hist3 = new_store_df['Produktverkäufe']
store_hist3.plot(kind='hist', ax=ax3_zwei, color='orangered', bins=30)
ax3_zwei.tick_params(labelrotation=0)
ax3_zwei.set_title('Produktverkäufe')

store_hist4 = staff_df['Tage_zwischen']
store_hist4.plot(kind='hist', ax=ax4_zwei, color='limegreen', bins=30)
ax4_zwei.tick_params(labelrotation=0)
ax4_zwei.set_title('Arbeitstagen')
plt.tight_layout()
# plt.show()

# Part 6: Correlation matrix
staff_df_new = staff_df.drop(['ID Nummer', 'Vorname', 'Nachname', 'Schichtverfügbarkeit', 'Qualifikationen',
                              'Gemischtwarenladen'], axis=1)
final_staff = pd.concat([staff_df_new, dum_uno, dum_dos, dum_tres], axis=1)
corr_matrix = final_staff.corr()


# corr_matrix.to_csv('corr_matrix_v2.csv', index=True, header=True)


def plot_line_graphs(dataframe):
    dataframe['Jahr'] = dataframe['Einstellungsdatum'].dt.year
    dataframe['Monat'] = dataframe['Einstellungsdatum'].dt.month
    pivot_tabelle = pd.pivot_table(dataframe, values='Tage_zwischen', index='Monat', columns='Jahr', aggfunc='mean')

    fig, nx = plt.subplots()
    for jahr in pivot_tabelle.columns:
        nx.plot(pivot_tabelle.index, pivot_tabelle[jahr], label=jahr, linestyle='--', marker='8')
        nx.grid(True, which='both')

    nx.set_xlabel('Monaten')
    nx.set_ylabel('Durchschnittliche Arbeitstage')
    nx.set_title('Vergleich der Arbeitstagdaten pro Jahr')
    nx.legend()
    # plt.show()


# Part 7: Assignment of workers to stores (1 to 8)
staff_df['Gemischtwarenladen'] = staff_df['Gemischtwarenladen'].astype(str).str.strip()
new_store_df['Gemischtwarenladen'] = new_store_df['Gemischtwarenladen'].astype(str).str.strip()
grouped_store = dict(tuple(new_store_df.groupby('Gemischtwarenladen')))
grouped_staff = dict(tuple(staff_df.groupby('Gemischtwarenladen')))
group_uno = grouped_staff["1"]
group_dos = grouped_staff["2"]
group_tres = grouped_staff["3"]
group_cuatro = grouped_staff["4"]
group_cinco = grouped_staff["5"]
group_seis = grouped_staff["6"]
group_siete = grouped_staff["7"]
group_ocho = grouped_staff["8"]
plot_line_graphs(group_cuatro)
plot_line_graphs(group_ocho)

# PART 8: Model Building - Data Preparation
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)

print("\nStaff data types:\n", staff_model.dtypes)
print("\nStore data types:\n", store_model.dtypes)

date_filter = store_model[store_model.Datum.dt.weekday == 6]
date_filter = date_filter[date_filter["Feiertag"] != 1]

date_range = pd.date_range(start="2015-01-01", end="2015-01-10", freq="D")
for date in date_range:
    staff_df[date.strftime('%d.%m.%Y')] = staff_df.apply(
        lambda row: 1 if row['Einstellungsdatum'] <= date <= row['Entlassungsdatum'] else 0, axis=1
    )

datum = date_range.strftime('%d.%m.%Y').tolist()
schichtverfugbarkeit = staff_df['Schichtverfügbarkeit'].unique().tolist()
qualifikationen = staff_df['Qualifikationen'].unique().tolist()
personal = staff_df['ID Nummer'].tolist()

qual_dict = staff_df.set_index('ID Nummer')['Qualifikationen'].to_dict()
pref_dict = staff_df.set_index('ID Nummer')['Schichtpräferenz'].map({'JA': 1, 'NEIN': 0}).to_dict()
store_lookup = staff_df.set_index('ID Nummer')['Gemischtwarenladen'].to_dict()
staff_active = {e: staff_df.loc[staff_df['ID Nummer'] == e].squeeze() for e in personal}

# PART 9: Model Building - Linear Programming Optimization Model
start = time.time()

# Decision variable: x[(employee, shift, date)]
x = {}
for e in personal:
    for d in datum:
        if staff_active[e][d] == 1:
            for s in schichtverfugbarkeit:
                if pref_dict.get(e, 1) == 1 or staff_active[e]['Schichtverfügbarkeit'] == s:
                    x[(e, s, d)] = plp.LpVariable(f"x_{e}_{s}_{d}", cat='Binary')

prob = plp.LpProblem("Employee_Scheduling", plp.LpMinimize)
prob += plp.lpSum(x.values())

for d in datum:
    for s in schichtverfugbarkeit:
        for store in staff_df['Gemischtwarenladen'].unique():
            eligible = [
                e for e in personal
                if store_lookup.get(e) == store and (e, s, d) in x
            ]
            if eligible:
                prob += plp.lpSum(x[(e, s, d)] for e in eligible) >= 1

end = time.time()
print("End time: ", end - start)
print("Solving the scheduling problem...")
prob.solve()
print("Solver status:", plp.LpStatus[prob.status])
print("Objective value (total assigned shifts):", plp.value(prob.objective))

# Shift Fulfillment Report
print("\n--- Shift Fulfillment Report ---")
# Prepare results for all combinations, including unstaffed shifts
results_data = []
for d in datum:
    for s in schichtverfugbarkeit:
        for store in staff_df['Gemischtwarenladen'].unique():
            assigned_staff = [
                e for e in personal if store_lookup.get(e) == store and (e, s, d) in x and x[(e, s, d)].varValue == 1
            ]
            results_data.append({
                'Datum': d,
                'Schicht': s,
                'Gemischtwarenladen': store,
                'Anzahl Mitarbeitende': len(assigned_staff),
                'Quota Erfüllt': "JA" if len(assigned_staff) > 0 else "NEIN"
            })

# Convert to DataFrame and export
results = pd.DataFrame(results_data)
results.to_csv("p_zwei_results.csv", index=False)

# PART 10: Model Building - Data Preparation
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)

store_retention = store_df.copy(deep=True)
# print("\nStore data types:\n", store_df.dtypes)

print("Datenrahmenform für alle Mitarbeiter:", staff_df.shape)
date_filter = store_retention[store_retention.Datum.dt.weekday == 6]
date_filter = date_filter[date_filter["Feiertag"] != 1]
date_list = list(date_filter["Datum"])
date_list_str = [date.strftime('%Y-%m-%d %H:%M:%S') for date in date_list]
date_list = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y') for date in date_list_str]

staff_retention['date_range'] = staff_retention.apply(lambda row: pd.date_range(start=row['Einstellungsdatum'],
                                                                  end=row['Entlassungsdatum'], freq='D'), axis=1)
datumsbereich = pd.date_range(start="01/01/2015", end="31/03/2023", freq='D')
for date in datumsbereich:
    staff_retention[date.strftime('%d.%m.%Y')] = staff_retention['date_range'].apply(lambda x: 1 if date in x else 0)

staff_retention.drop('date_range', axis=1, inplace=True)
print("Datenrahmenform für alle Mitarbeiter:", staff_retention.shape)
staff_retention = staff_retention.drop(columns=date_list)
print("Datenrahmenform für alle Mitarbeiter:", staff_retention.shape)

# PART 11: Model Building - Retention analysis Logistics Regression Model
dumm_eins = pd.get_dummies(staff_retention["Schichtverfügbarkeit"], prefix="Schichtverfügbarkeit")
dumm_zwei = pd.get_dummies(staff_retention["Qualifikationen"], prefix="Qualifikationen")
dumm_drei = pd.get_dummies(staff_retention["Gemischtwarenladen"], prefix="Gemischtwarenladen")

df_logit = pd.concat([staff_retention, dumm_eins, dumm_zwei, dumm_drei], axis=1)
predictors = ["Überstunden", "Schichtpräferenz", "Schichtverfügbarkeit_Morgenschicht",
              "Schichtverfügbarkeit_Nachmittagsschicht", "Qualifikationen_Aushilfe", "Qualifikationen_Reinigungskraft",
              "Qualifikationen_Verkäufer", "Gemischtwarenladen_1", "Gemischtwarenladen_2", "Gemischtwarenladen_3",
              "Gemischtwarenladen_4", "Gemischtwarenladen_5", "Gemischtwarenladen_6", "Gemischtwarenladen_7",
              "Gemischtwarenladen_8"]
mean_resp = df_logit["Tage_zwischen"].mean()
df_logit["response"] = (df_logit["Tage_zwischen"] < mean_resp).astype(int)
print("\n\nDF_LOGIT data types:\n", df_logit.dtypes)
response = ["response"]

X_train, X_test, y_train, y_test = train_test_split(df_logit[predictors], df_logit[response], train_size=0.8,
                                                    random_state=0)
model = sm.Logit(y_train, X_train).fit()
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# PART 11: Model Building - Accuracy and precision tests
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print("\n\n\n", model.summary())
