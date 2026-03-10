# Household Energy Consumption Forecasting
### *Comparing Statistical (SARIMAX) vs. Deep Learning (LSTM) Approaches*

## Project Overview
This project predicts hourly electricity consumption for a single household in France. Using 4 years of historical data, I built and compared a traditional statistical model (SARIMAX) against a Deep Learning architecture (LSTM) to determine which better handles the "spiky," non-linear nature of residential power demand.

## Key Results
| Model | MAE (kW) | RMSE (kW) | $R^2$ |
| :--- | :--- | :--- | :--- |
| **LSTM (t+24)** | **0.4450** | **0.6129** | **0.2998** |
| **Seasonal Naive** | 0.5219 | 0.7803 | -0.1431 |
| **SARIMAX** | 0.6068 | 0.8007 | -0.2037 |

**The Verdict**: The **LSTM** outperformed the baseline by **26%**, successfully capturing complex appliance interactions that the linear SARIMAX model missed.

## Modeling Approach

### 1. SARIMAX (Statistical)
Configured as **(2,1,2) x (1,1,1,24)** with **Fourier Terms (K=5)** to capture daily and weekly seasonality.
* **The "Chef" Analogy**: This model acts like a chef following a strict recipe—great for stable trends but struggled with the chaotic "spikes" of a single home.

### 2. LSTM (Deep Learning)
A recurrent neural network designed for long-term memory.
* **Non-Linear Power**: Won because it learned the "hidden" relationships between voltage, intensity, and time that linear models cannot see.

##  Data Source
The dataset used is the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository 
(https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). Due to its size (~130MB), the raw data is not included in this repository.

## Business Impact
* **Grid Stability**: Precise day-ahead planning for utility providers.
* **Renewable Integration**: Better battery storage sizing to bridge the demand gap.
