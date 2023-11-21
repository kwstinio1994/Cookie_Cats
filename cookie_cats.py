import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

# Title and Introduction
st.title('Cookie Cats Player Engagement Analysis')
st.write('''
         This application analyzes the player engagement data from the Cookie Cats game. 
         It focuses on player retention, game rounds played, and the impact of different game versions.
         Dataset Source: https://www.kaggle.com/datasets/arpitdw/cokie-cats
         ''')

# Load data
@st.cache
def load_data():
    data = pd.read_csv('cookie_cats.csv')
    return data

cookie_cats_data = load_data()

# Basic Data Quality Check
st.header('Data Quality Check')
st.subheader('Basic Data Overview')
if cookie_cats_data.isnull().sum().sum() == 0:
    st.write('No missing values found in the dataset.')
else:
    st.write('Missing values found in the dataset:', cookie_cats_data.isnull().sum())

if cookie_cats_data.duplicated().sum() == 0:
    st.write('No duplicate rows found in the dataset.')
else:
    st.write('Duplicate rows found in the dataset:', cookie_cats_data.duplicated().sum())

# Basic Statistical Analysis
st.header('Basic Statistical Analysis')
st.write('Descriptive Statistics of the Dataset (Overall):')
# Calculate the sum of sum_gamerounds directly
total_gamerounds = cookie_cats_data['sum_gamerounds'].sum()
# Display the result
st.write('Total sum of game rounds played by all users:', total_gamerounds)

st.subheader('Key Metrics')
total_players = cookie_cats_data.shape[0]
average_game_rounds = cookie_cats_data['sum_gamerounds'].mean()
retention_1_day_rate = cookie_cats_data['retention_1'].mean() * 100
retention_7_days_rate = cookie_cats_data['retention_7'].mean() * 100

st.write(f'Total Number of Players: {total_players}')
st.write(f'Average Game Rounds per Player: {average_game_rounds:.2f}')
st.write(f'1-Day Retention Rate: {retention_1_day_rate:.2f}%')
st.write(f'7-Day Retention Rate: {retention_7_days_rate:.2f}%')

# Descriptive Statistics by Game Version vol
st.subheader('Descriptive Statistics by Game Version')
for version in cookie_cats_data['version'].unique():
    st.write(f'For {version}:')
    version_data = cookie_cats_data[cookie_cats_data['version'] == version]
    total_users = version_data['userid'].nunique()
    st.write('Total number of users:', total_users)
    rounds_stats = version_data['sum_gamerounds'].describe()
    st.write(rounds_stats)
    retention_1_day_rate = version_data['retention_1'].mean()
    retention_7_day_rate = version_data['retention_7'].mean()
    st.write(f'Mean 1-day retention rate: {retention_1_day_rate:.2%}')
    st.write(f'Mean 7-day retention rate: {retention_7_day_rate:.2%}')


# Initial Visualizations
st.header('Initial Data Visualizations')
st.subheader('Distribution of Game Rounds Played by Game Version')

# Determine a reasonable maximum value for the x-axis, such as the 95th percentile of game rounds
max_gamerounds = int(cookie_cats_data['sum_gamerounds'].quantile(0.95))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i, version in enumerate(cookie_cats_data['version'].unique()):
    # Filter the data for the version and rounds less than the 95th percentile
    data = cookie_cats_data[(cookie_cats_data['version'] == version) & 
                            (cookie_cats_data['sum_gamerounds'] <= max_gamerounds)]
    sns.histplot(data['sum_gamerounds'], bins=30, kde=False, ax=ax[i])
    ax[i].set_title(f'Distribution for {version}')
    ax[i].set_xlim(0, max_gamerounds)
    ax[i].set_xlabel('Game Rounds Played')
    ax[i].set_ylabel('Count')
st.pyplot(fig)


# Exploratory Data Analysis (EDA)
st.header('Exploratory Data Analysis')
st.subheader('Retention Rates by Game Version')
# Plotting retention rates by version
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(x='version', y='retention_1', data=cookie_cats_data, ax=ax[0])
ax[0].set_title('1-Day Retention Rate by Game Version')
sns.barplot(x='version', y='retention_7', data=cookie_cats_data, ax=ax[1])
ax[1].set_title('7-Day Retention Rate by Game Version')
st.pyplot(fig)


# A/B Testing Analysis
st.header('A/B Testing Analysis')

# Define a function to calculate the effect size (Cohen's d)
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)

# 1-Day Retention Analysis
st.subheader('1-Day Retention')
retention_1_day_gate_30 = cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['retention_1']
retention_1_day_gate_40 = cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['retention_1']
t_stat_1_day, p_val_1_day = stats.ttest_ind(retention_1_day_gate_30, retention_1_day_gate_40)
d_1_day = cohen_d(retention_1_day_gate_30, retention_1_day_gate_40)
st.write(f'p-value: {p_val_1_day:.4f}')
st.write(f'Cohen\'s d: {d_1_day:.4f}')
st.write('Conclusion: The difference in 1-day retention between the two versions is ' + 
         ('statistically significant.' if p_val_1_day < 0.05 else 'not statistically significant.'))

# 7-Day Retention Analysis
st.subheader('7-Day Retention')
retention_7_day_gate_30 = cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['retention_7']
retention_7_day_gate_40 = cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['retention_7']
t_stat_7_day, p_val_7_day = stats.ttest_ind(retention_7_day_gate_30, retention_7_day_gate_40)
d_7_day = cohen_d(retention_7_day_gate_30, retention_7_day_gate_40)
st.write(f'p-value: {p_val_7_day:.4f}')
st.write(f'Cohen\'s d: {d_7_day:.4f}')
st.write('Conclusion: The difference in 7-day retention between the two versions is ' + 
         ('statistically significant.' if p_val_7_day < 0.05 else 'not statistically significant.'))

# Sum of Game Rounds Played Analysis
st.subheader('Sum of Game Rounds Played')
sum_gamerounds_gate_30 = cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['sum_gamerounds']
sum_gamerounds_gate_40 = cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['sum_gamerounds']
t_stat_gamerounds, p_val_gamerounds = stats.ttest_ind(sum_gamerounds_gate_30, sum_gamerounds_gate_40)
d_gamerounds = cohen_d(sum_gamerounds_gate_30, sum_gamerounds_gate_40)
st.write(f'p-value: {p_val_gamerounds:.4f}')
st.write(f'Cohen\'s d: {d_gamerounds:.4f}')
st.write('Conclusion: The difference in the sum of game rounds played between the two versions is ' + 
         ('statistically significant.' if p_val_gamerounds < 0.05 else 'not statistically significant.'))

# Statistical Testing
st.header('Statistical Testing')
# T-test for retention rates between different versions
retention_1_gate_30 = cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['retention_1']
retention_1_gate_40 = cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['retention_1']
t_stat, p_val = ttest_ind(retention_1_gate_30, retention_1_gate_40)
st.write('1-Day Retention T-Test Results: T-statistic =', t_stat, ', P-value =', p_val)

# Explanation based on p-value
if p_val < 0.05:
    st.write('''
             The difference in 1-day retention between the two game versions is statistically significant.
             This suggests that the game version has a significant impact on player retention.
             ''')
else:
    st.write('''
             The difference in 1-day retention between the two game versions is not statistically significant.
             This suggests that the game version might not have a significant impact on player retention.
             ''')

# Enhanced A/B Testing Analysis
st.header('Enchanced A/B Testing Analysis using Bootstrap')
# Bootstrap sampling function
def bootstrap_sampling(data, n_bootstrap=1000):
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bootstrap_sample = data.sample(frac=1, replace=True)
        bootstrap_means[i] = bootstrap_sample.mean()
    return bootstrap_means

# Creating 1000 samples (bootstraping) for 1-day & 7-days retentions
bootstrap_1_day_30 = bootstrap_sampling(cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['retention_1'])
bootstrap_1_day_40 = bootstrap_sampling(cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['retention_1'])
bootstrap_7_days_30 = bootstrap_sampling(cookie_cats_data[cookie_cats_data['version'] == 'gate_30']['retention_7'])
bootstrap_7_days_40 = bootstrap_sampling(cookie_cats_data[cookie_cats_data['version'] == 'gate_40']['retention_7'])

# Plotting the bootstrap distributions as line plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.kdeplot(bootstrap_1_day_30, shade=True, color="blue", ax=ax[0], label='Gate 30')
sns.kdeplot(bootstrap_1_day_40, shade=True, color="red", ax=ax[0], label='Gate 40')
ax[0].set_title('Bootstrap Distribution for 1-Day Retention')
ax[0].legend()
sns.kdeplot(bootstrap_7_days_30, shade=True, color="blue", ax=ax[1], label='Gate 30')
sns.kdeplot(bootstrap_7_days_40, shade=True, color="red", ax=ax[1], label='Gate 40')
ax[1].set_title('Bootstrap Distribution for 7-Days Retention')
ax[1].legend()
st.pyplot(fig)

# Calculating retention rate differences
diff_retention_1_day = bootstrap_1_day_30.mean() - bootstrap_1_day_40.mean()
diff_retention_7_days = bootstrap_7_days_30.mean() - bootstrap_7_days_40.mean()
st.write(f'Difference in 1-Day Retention Rate: {diff_retention_1_day:.2%}')
st.write(f'Difference in 7-Days Retention Rate: {diff_retention_7_days:.2%}')

# Probability calculation
prob_1_day = np.mean(bootstrap_1_day_30 > bootstrap_1_day_40)
prob_7_days = np.mean(bootstrap_7_days_30 > bootstrap_7_days_40)
st.write(f'Probability of higher 1-day retention when gate is at level 30: {prob_1_day:.2%}')
st.write(f'Probability of higher 7-days retention when gate is at level 30: {prob_7_days:.2%}')

# Evaluating results and making recommendation
st.subheader('Evaluation and Recommendation')
if prob_7_days > 0.5:
    st.write('''
             Based on the A/B testing analysis, it appears more likely for players to retain longer when the gate is at level 30.
             We recommend keeping or experimenting further with the gate at level 30 to improve 7-days retention rates.
             ''')
else:
    st.write('''
             The results do not conclusively favor the gate at level 30 for 7-days retention.
             Further analysis or different A/B testing approaches may be required for a more definitive recommendation.
             ''')

# Predictive Modeling
st.header('Predictive Modeling')
# Preparing data for modeling
X = cookie_cats_data[['sum_gamerounds', 'version']]
y = cookie_cats_data['retention_7']
# Encoding categorical variable
X = pd.get_dummies(X, drop_first=True)
# Splitting the data (80% train, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)
st.write('Model Performance Metrics:')
st.text(classification_report(y_test, y_pred))

# User Input for Prediction
st.header('User Input for Retention Prediction')
user_input = st.number_input('Enter the number of game rounds played:', min_value=0)
version_input = st.selectbox('Select the game version:', ['gate_30', 'gate_40'])
# Encoding version_input
encoded_input = 1 if version_input == 'gate_40' else 0
# Predicting
if st.button('Predict Retention'):
    prediction = model.predict_proba([[user_input, encoded_input]])[0][1]
    st.write(f'Probability of returning after 7 days: {prediction:.2f}')

# Interactive User Features
st.header('Interactive User Features')
# Slider for filtering data
slider_val = st.slider('Filter data by sum of game rounds played', min_value=int(cookie_cats_data['sum_gamerounds'].min()), max_value=int(cookie_cats_data['sum_gamerounds'].max()))
filtered_data = cookie_cats_data[cookie_cats_data['sum_gamerounds'] <= slider_val]
st.write(filtered_data)

# Data Quality Enhancements
st.header('Data Quality Enhancements')
st.subheader('Outlier Analysis for Game Rounds Played')
# Create a log-transformed boxplot
fig, ax = plt.subplots()
sns.boxplot(x=np.log1p(cookie_cats_data['sum_gamerounds']), ax=ax)  # log1p is used to handle 0 values in the data
ax.set_xlabel('Log of Game Rounds Played')
st.pyplot(fig)


# Conclusions and Recommendations
st.header('Conclusions and Recommendations')
st.write('''
         Based on our analysis of the Cookie Cats data, we have observed several key insights:
         1. There is a statistically significant difference in 1-day and 7-day retention rates between the two game versions (gate_30 and gate_40).
         2. The total number of game rounds played shows a wide distribution among players, indicating varied engagement levels.
         3. The predictive model suggests that the number of game rounds played is a significant predictor of 7-day retention.

         Recommendations:
         - Consider modifying game features in the version with lower retention rates to improve player engagement.
         - Focus on strategies to increase the total number of game rounds played by new players, as it correlates with higher retention.
         - Continue collecting and analyzing data to refine understanding of player behavior and preferences.

         Please note that this analysis is exploratory and based on a limited dataset. It should be used as a starting point for deeper investigation and not as a definitive guide for game development decisions.
     
        ''')

# About/Contact Section
st.header('About/Contact')
st.write('''
         This analysis was conducted by Konstantinos Doulkeridis. 
         For more information or to contact, please visit https://www.linkedin.com/in/konstantinos-doulkeridis/ or email at konstantinos.doulkeridis@hotmail.com.
         ''')

