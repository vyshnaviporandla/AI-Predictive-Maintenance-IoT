# Project Architecture

## Overview
The project follows a simple predictive maintenance workflow used in industrial monitoring systems. It begins with simulated sensor readings from IoT devices and ends with a failure risk prediction and alert signal.

## Data Flow
1. IoT sensors generate temperature, vibration, pressure, RPM, current, and humidity values.
2. The data is stored in CSV format.
3. The preprocessing stage cleans values, removes duplicates, and creates engineered features.
4. A machine learning model learns the relationship between sensor signals and machine failure events.
5. The model predicts whether a device is likely to fail soon.
6. A warning is raised when the predicted risk crosses a threshold.
7. The results are visualized in charts and saved to the output folders.

## Block Diagram
```text
IoT Sensors --> CSV Data --> Data Cleaning --> Feature Engineering --> Model Training --> Prediction --> Alert + Visualization
```

## Module Summary
- data_generator.py: creates realistic synthetic IoT sensor records
- preprocessing.py: cleans and transforms the dataset
- model_building.py: trains and evaluates the predictive model
- visualization.py: generates visual diagnostics and charts
- main.py: runs the full workflow end-to-end

## Expected Outcome
A beginner-friendly machine learning project that mimics an industrial predictive maintenance pipeline while remaining easy to run on a student laptop.
