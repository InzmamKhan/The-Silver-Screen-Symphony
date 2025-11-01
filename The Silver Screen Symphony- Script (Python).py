import pandas as pd # type: ignore
import numpy as np # type: ignore

# Assuming the data is saved as 'movies_data.csv'
data = pd.read_csv("C:\\Users\\khani\\OneDrive\\Desktop\\Digital Relics\\The Silver Screen Symphony\\The Silver Screen Symphony- Source - Original - CSV.csv")
df = pd.DataFrame(data)

# --- Cleaning Steps ---

# 1. Clean 'Gross' column: Remove commas and convert to numeric (integer)
df['Gross'] = df['Gross'].str.replace(',', '', regex=False)
# Handle potential missing or non-numeric values by converting to numeric, coercing errors to NaN
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce').fillna(0).astype(np.int64) # Fill NaN with 0 or a more appropriate value

# 2. Clean 'Runtime' column: Remove ' min' and convert to numeric (integer)
df['Runtime (min)'] = df['Runtime'].str.replace(' min', '', regex=False)
df['Runtime (min)'] = pd.to_numeric(df['Runtime (min)'], errors='coerce').fillna(0).astype(int) # Fill NaN with 0
df.drop('Runtime', axis=1, inplace=True)

# 3. Clean 'Released_Year' column: Convert to numeric, coercing non-numeric values (like 'PG-13' for a year) to NaN
# Then, fill NaNs with a reasonable strategy (e.g., median or 0, but 0 is safer for now) and convert to integer
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
df['Released_Year'] = df['Released_Year'].fillna(df['Released_Year'].median() if not df['Released_Year'].isnull().all() else 0).astype(int)

# 4. Handle remaining missing values (NaNs) in numerical columns 'Meta_score' and 'IMDB_Rating'
# Filling with the mean or median is a common strategy
df['Meta_score'].fillna(df['Meta_score'].mean(), inplace=True)
df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean(), inplace=True)

# 5. Drop the 'Poster_Link' column as it's not needed for analysis
df.drop('Poster_Link', axis=1, inplace=True)

print("Data Loading and Cleaning Complete.")
print("\nFirst 5 rows of the cleaned data:")
print(df.head())

# Assuming the data frame 'df' is cleaned and 'Gross' is now numerical
# Use Pearson correlation as it's the standard for numerical variables
correlation = df['IMDB_Rating'].corr(df['Gross'])

print(f"Pearson Correlation between IMDB Rating and Gross Revenue: {correlation:.4f}")

# Interpretation (optional)
if abs(correlation) > 0.5:
    strength = "strong"
elif abs(correlation) > 0.3:
    strength = "moderate"
elif abs(correlation) > 0.1:
    strength = "weak"
else:
    strength = "very weak/negligible"

direction = "positive" if correlation > 0 else "negative"

print(f"This indicates a {strength} {direction} linear relationship.")

# Group by 'Certificate' and calculate the mean IMDB Rating and the sum of Gross
grouped_data = df.groupby('Certificate').agg(
    Average_IMDB_Rating=('IMDB_Rating', 'mean'),
    Total_Gross=('Gross', 'sum'),
    Count_of_Movies=('Series_Title', 'count')
).reset_index()

# Sort by Average IMDB Rating for better comparison
grouped_data_sorted = grouped_data.sort_values(by='Average_IMDB_Rating', ascending=False)

print("Grouped Data by Certificate:")
print(grouped_data_sorted)

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Set a style for better visualization
sns.set_style("whitegrid")

plt.figure(figsize=(8, 5))
# Using seaborn for a clean scatter plot
sns.scatterplot(
    x='Gross',
    y='IMDB_Rating',
    data=df,
    hue='Certificate', # Optional: Add color coding by Certificate
    palette='viridis',
    s=100, # Size of the dots
    alpha=0.7 # Transparency
)

# Add a title and labels
plt.title('Scatter Plot of IMDB Rating vs. Gross Revenue ', fontsize=16)
plt.xlabel('Gross Revenue (in millions)', fontsize=12)
plt.ylabel('IMDB Rating', fontsize=12)

# Optional: Improve x-axis readability by formatting as millions
# Note: For simplicity, the label indicates millions, but the raw numbers are shown.
# A more advanced formatting function would be needed to truly display in 'M'.

plt.tight_layout() # Adjust plot to fit all elements
plt.show()

# Prepare data (reuse the grouped data from step 3)
certificate_avg_rating = df.groupby('Certificate')['IMDB_Rating'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8, 5))

# Create the bar plot
sns.barplot(
    x='Certificate',
    y='IMDB_Rating',
    data=certificate_avg_rating,
    palette='magma'
)

# Add labels and title
plt.title('Average IMDB Rating by Movie Certificate ', fontsize=16)
plt.xlabel('Certificate', fontsize=12)
plt.ylabel('Average IMDB Rating', fontsize=12)

# Annotate bars with the average rating
for index, row in certificate_avg_rating.iterrows():
    plt.text(index, row['IMDB_Rating'], f'{row["IMDB_Rating"]:.2f}', color='black', ha="center")

plt.tight_layout()
plt.show()

# Identify all numerical columns
numerical_cols = ['Released_Year', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross', 'Runtime (min)']

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 5))

# Create the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,         # Show the correlation values on the heatmap
    cmap='coolwarm',    # Color map
    fmt=".2f",          # Format the numbers to two decimal places
    linewidths=.5,      # Lines between cells
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title('Heatmap Correlation Matrix of Numerical Variables ', fontsize=16)
plt.tight_layout()
plt.show()