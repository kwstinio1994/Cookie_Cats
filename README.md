# Cookie Cats Player Engagement Analysis

## Description
This Streamlit application provides an in-depth analysis of player engagement in the mobile game Cookie Cats. The application delves into player retention, the number of game rounds played, and the variations in engagement across different game versions. 
The project involves data quality checks, descriptive statistics, hypothesis testing, predictive modeling, and visualization of results.

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
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/3296354a-4e07-4b89-b955-5c4343d4ebaa)
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/57b16a38-2eb2-4ebc-8648-fc6bf0fa11d2)
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/3cce33b0-f3a8-469b-a09b-24ae02527ac5)

### Initial Data Visualizations
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/82db7749-2e51-4712-b555-0887f3a00534)
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/763bfb62-e0d4-4db3-abc8-e9a4048359c1)

## A/B Testing Analysis
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/a646ac0c-20c3-4e4b-b3a3-8fe295dee2d9)
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/895d16ab-93c6-463f-8569-ef295d1fcb48)

## Enchanced A/B Testing Analysis using Bootstrap.
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/849df56a-124e-4f21-a6c4-58a3edf396b1)

## Predictive Modeling
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/4ea1f612-303b-4a80-b572-6d0d483193ff)

## User Input for Retention Prediction
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/241a7eca-8d20-4215-80d0-61ec13daea19)

You can add enter the number of the played game rounds and the game version and get the Probability of returning after 7 days

Example:
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/6e9594e6-830a-4b5a-9e80-e578d4a09f91)

## Interactive User Features
Ability to filter data by sum of game rounds played
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/83facd3a-9c2a-4cf9-9c82-4abf9ac0d4b1)

## Data Quality Enhancements
![image](https://github.com/kwstinio1994/Cookie_Cats/assets/151637921/dba58842-e183-4034-a7a0-5269ac16f512)

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




