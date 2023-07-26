from scipy.stats import chi2_contingency, pointbiserialr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function for plotting the data
def plot_data(df, column, kind='bar', color='skyblue', figsize=(10,5)):
    df[column].value_counts().sort_values().plot(kind=kind, color=color, figsize=figsize)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# Function for performing Chi-square test and displaying the results
def chi_square_test(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, ex = chi2_contingency(crosstab)
    print(f"\n{col1} and {col2}")
    print(f"Chi-square statistic: {chi2}, p-value: {p}")

# data read
df = pd.read_csv("./dataset/metadata.csv")

# Basic Information
print("The dataset has", df.shape[0], "rows and", df.shape[1], "columns.\n")
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(include='all'), "\n")

# Drop missing values
df = df.dropna()

# Univariate Analysis
features = ['lesion_type', 'age', 'confirmation', 'sex', 'localization']
for feature in features:
    if feature == 'age':
        plot_data(df, feature, figsize=(12, 6))
    else:
        plot_data(df, feature)

# Correlations between attributes
print("Correlations:\n")
pairs = [('confirmation', 'lesion_type'), 
         ('sex', 'lesion_type'), 
         ('localization', 'lesion_type'), 
         (('confirmation', 'sex'), 'localization')]

for pair in pairs:
    if isinstance(pair[0], tuple):
        cross_tab = pd.crosstab([df[col] for col in pair[0]], df[pair[1]])
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        print(f"\n{pair[0]} and {pair[1]}")
        print(f"Chi-square statistic: {chi2}, p-value: {p}")
    else:
        chi_square_test(df, *pair)

# Box plot
sns.boxplot(x='lesion_type', y='age', data=df)
plt.show()

# Point-biserial correlation (assuming lesion_type is binary)
res = pointbiserialr(df['lesion_type'], df['age'])
print("\nLesion_type and age")
print(res)