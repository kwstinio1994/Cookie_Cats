# Cookie Cats Player Engagement Analysis

## Description
This Streamlit application provides an in-depth analysis of player engagement in the mobile game Cookie Cats. The application delves into player retention, the number of game rounds played, and the variations in engagement across different game versions. 
The project involves data quality checks, descriptive statistics, hypothesis testing, predictive modeling, and visualization of results. You can download the repository and run it directly or you can visit https://cookiecats-rkhl25hdzmyqqrx6ucyaay.streamlit.app/ and access this web app using your credentials.

## Installation
To use this application, Python must be installed on your system. Follow these steps:

bash
git clone https://github.com/kwstinio1994/Cookie_Cats.git
cd Cookie_Cats
pip install -r requirements.txt

## Usage
Start the Streamlit application with this command:

streamlit run cookie_cats.py

# Results

## Data Quality Key
### Basic Data Overview
No missing values found in the dataset.
No duplicate rows found in the dataset.

### Basic Data Overview
Total Number of Players: 90189

Average Game Rounds per Player: 51.87

1-Day Retention Rate: 44.52%

7-Day Retention Rate: 18.61%

### Descriptive Statistics by Game Version
![cookie_cats_1](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/0ef80e4b-7e4f-414d-9cd5-dbb14ffa82fd)
![cookie_cats_2](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/3a1946ec-ee24-4a01-944e-0f3823efbc6b)

## Initial Data Visualizations

### Distribution of Game Rounds Played by Game Version
![cookie_cats_3](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/1088da98-b9a7-4d3e-a502-51df7aee905d)

### Exploratory Data Analysis

![cookie_cats_4](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/1975d1f3-67c1-42f0-a9c1-019ec797b844)

## A/B Testing Analysis
### 1-Day Retention
p-value: 0.0744

Cohen's d: 0.0119

Conclusion: The difference in 1-day retention between the two versions is not statistically significant.

### 7-Day Retention
p-value: 0.0016

Cohen's d: 0.0211

Conclusion: The difference in 7-day retention between the two versions is statistically significant.

### Sum of Game Rounds Played
p-value: 0.3729

Cohen's d: 0.0059

Conclusion: The difference in the sum of game rounds played between the two versions is not statistically significant.

## Statistical Testing
1-Day Retention T-Test Results: 
T-statistic = 1.7840979256519656
P-value = 0.07441111525563184

The difference in 1-day retention between the two game versions is not statistically significant. This suggests that the game version might not have a significant impact on player retention.

## Enchanced A/B Testing Analysis using Bootstrap.
![cookie_cats_5](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/e166da64-1619-49b8-b571-0fc4224c49a7)

Difference in 1-Day Retention Rate: 0.59%

Difference in 7-Days Retention Rate: 0.82%

Probability of higher 1-day retention when gate is at level 30: 95.80%

Probability of higher 7-days retention when gate is at level 30: 99.90%

## Evaluation and Recommendation
Based on the A/B testing analysis, it appears more likely for players to retain longer when the gate is at level 30. 
We recommend keeping or experimenting further with the gate at level 30 to improve 7-days retention rates.

## Predictive Modeling
![cookie_cats_6](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/96671ef0-aae8-4752-a324-dfcde1cc7b3a)

## User Input for Retention Prediction
![cookie_cats_7](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/64e6180a-0fd8-438a-bedd-56770753a0f0)

You can add enter the number of the played game rounds and the game version and get the Probability of returning after 7 days

Example:
![cookie_cats_8](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/886c1420-6485-4586-89c3-166a8b90dc88)

## Interactive User Features
Ability to filter data by sum of game rounds played
![cookie_cats_9](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/8b2ca4ca-4060-4c71-aa5a-b47e34aacf21)

## Data Quality Enhancements
![cookie_cats_10](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/6275c58a-7481-4337-9308-3f77d4e5cd4f)

## Conclusions and Recommendations
Based on our analysis of the Cookie Cats data, we have observed several key insights:

1. There is a statistically significant difference in 1-day and 7-day retention rates between the two game versions (gate_30 and gate_40).
2. The total number of game rounds played shows a wide distribution among players, indicating varied engagement levels.
3. The predictive model suggests that the number of game rounds played is a significant predictor of 7-day retention.

## Recommendations:

a. Consider modifying game features in the version with lower retention rates to improve player engagement.
b. Focus on strategies to increase the total number of game rounds played by new players, as it correlates with higher retention.
c. Continue collecting and analyzing data to refine understanding of player behavior and preferences.

Please note that this analysis is exploratory and based on a limited dataset. It should be used as a starting point for deeper investigation and not as a definitive guide for game development decisions.

## About/Contact
This analysis was conducted by Konstantinos Doulkeridis. For more information or to contact,
please visit https://www.linkedin.com/in/konstantinos-doulkeridis/ or email at konstantinos.doulkeridis@hotmail.com.

## Contributing
While this is a relatively small project and open contributions are not available, 
I am open to discussions for improvements, collaborations, or contributions to other projects. 
Feel free to reach out directly for such discussions.

## Acknowledgments
Data Source: https://www.kaggle.com/datasets/arpitdw/cokie-cats 




