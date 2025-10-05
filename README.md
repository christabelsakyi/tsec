# Energy Consumption Time Series Forecasting

This project implements a time series forecasting solution for predicting electricity consumption based on historical data. It provides multiple model options, a Flask API for deployment, and visualization tools.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [License](#license)

## Project Overview

This forecasting system provides predictions for future electricity consumption based on historical data using various time series models such as ARIMA, Facebook Prophet, and LSTM neural networks. The application includes data preprocessing, model training, evaluation, and a RESTful API.

### Dataset

The project uses the [Household Electric Power Consumption Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) or the cleaned version available on [Kaggle (AEP_hourly.csv)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

## Features

- **Data Preprocessing**: Automated cleaning, resampling, and feature engineering
- **Multiple Models**: 
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Facebook Prophet
  - LSTM (Long Short-Term Memory) neural networks
- **Model Evaluation**: Visualizations and metrics (MAE, RMSE)
- **Flask API**: RESTful endpoints for forecasting
- **Visualization**: Interactive charts for trends and predictions
- **Containerization**: Docker support for easy deployment

## Installation

### Prerequisites

- Python 3.8+
- pip
- Docker (optional)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-consumption-forecasting.git
   cd energy-consumption-forecasting
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```bash
   mkdir -p data/raw
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip -O data/raw/household_power_consumption.zip
   unzip data/raw/household_power_consumption.zip -d data/raw
   ```
   Alternatively, download the [AEP_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) from Kaggle and place it in the `data/raw` directory.

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t energy-forecasting .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 energy-forecasting
   ```

## Usage

### Data Preprocessing

Run the preprocessing script to prepare your data:

```bash
python src/data/preprocessing.py
```

### Model Training

Train the forecasting model with:

```bash
python src/models/train_model.py --model prophet
```

Available model options: `arima`, `prophet`, `lstm`

### Running the API

Start the Flask API:

```bash
python run.py
```

Or using Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "src.api.app:app"
```

## API Documentation

### Endpoints

#### Health Check

```
GET /api/health
```

Response:
```json
{
  "status": "ok",
  "message": "Service is running"
}
```

#### Forecast

```
POST /api/forecast
```

Request Body:
```json
{
  "horizon": 24
}
```

Response:
```json
{
  "status": "success",
  "forecast": {
    "timestamps": ["2025-03-25 12:00:00", "2025-03-25 13:00:00", ...],
    "predictions": [345.6, 350.2, ...],
    "lower_bound": [335.1, 340.8, ...],
    "upper_bound": [356.1, 359.6, ...]
  },
  "model_type": "prophet"
}
```

## Model Details

### ARIMA

The ARIMA (AutoRegressive Integrated Moving Average) model captures temporal dependencies in time series data using autoregressive and moving average components with differencing for stationarity.

### Facebook Prophet

Prophet is designed for forecasting time series data with strong seasonal patterns and multiple seasonalities. It handles missing data and outliers well.

### LSTM

LSTM (Long Short-Term Memory) neural networks can capture complex temporal dependencies in the data and are particularly useful for time series with long-term patterns.

## License

This project is licensed under the MIT License - see the LICENSE file for details.