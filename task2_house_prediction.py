import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# For actual Kaggle dataset, use:
# df = pd.read_csv('house_prices.csv')

# Creating a realistic sample dataset
np.random.seed(42)
n_samples = 500

data = {
    'num_rooms': np.random.randint(2, 8, n_samples),
    'size_sqft': np.random.randint(800, 4000, n_samples),
    'location_score': np.random.uniform(1, 10, n_samples),  # 1-10 scale
    'age_years': np.random.randint(0, 50, n_samples),
    'num_bathrooms': np.random.randint(1, 5, n_samples),
    'has_garage': np.random.randint(0, 2, n_samples),  # 0 or 1
    'distance_to_city_km': np.random.uniform(1, 50, n_samples)
}

# Generate price based on features with some realistic relationships
df = pd.DataFrame(data)
df['price'] = (
    50000 +  # Base price
    df['num_rooms'] * 25000 +  # Price per room
    df['size_sqft'] * 120 +  # Price per sqft
    df['location_score'] * 30000 +  # Location premium
    df['num_bathrooms'] * 15000 +  # Bathroom value
    df['has_garage'] * 20000 +  # Garage value
    -df['age_years'] * 800 +  # Depreciation
    -df['distance_to_city_km'] * 500 +  # Distance penalty
    np.random.normal(0, 30000, n_samples)  # Random noise
)

# Ensure no negative prices
df['price'] = df['price'].clip(lower=100000)

print("=" * 70)
print("HOUSE PRICE PREDICTION MODEL - LINEAR REGRESSION")
print("=" * 70)

# DATA EXPLORATION
print("\n1. DATASET OVERVIEW:")
print(f"   Total samples: {len(df)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\n\n2. DATASET STATISTICS:")
print(df.describe())

print(f"\n\n3. MISSING VALUES:")
print(df.isnull().sum())

# PREPROCESSING
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: price")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Feature Scaling (optional but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING
print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n✓ Model trained successfully!")

# Display model coefficients
print("\n\nMODEL COEFFICIENTS:")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(coefficients)
print(f"\nIntercept: ${model.intercept_:,.2f}")

# MODEL EVALUATION
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTRAINING SET PERFORMANCE:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")

print("\nTEST SET PERFORMANCE:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")

print(f"\n  Interpretation:")
print(f"  - The model explains {test_r2*100:.2f}% of the variance in house prices")
print(f"  - Average prediction error: ${test_mae:,.2f}")

# VISUALIZATIONS
fig = plt.figure(figsize=(16, 10))

# 1. Actual vs Predicted (Test Set)
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)', fontweight='bold')
plt.ylabel('Predicted Price ($)', fontweight='bold')
plt.title(f'Actual vs Predicted Prices\nR² = {test_r2:.4f}', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Residual Plot
plt.subplot(2, 3, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($)', fontweight='bold')
plt.ylabel('Residuals ($)', fontweight='bold')
plt.title('Residual Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Feature Importance (Absolute Coefficients)
plt.subplot(2, 3, 3)
importance = coefficients.copy()
importance['Abs_Coefficient'] = importance['Coefficient'].abs()
importance = importance.sort_values('Abs_Coefficient', ascending=True)
plt.barh(importance['Feature'], importance['Abs_Coefficient'], color='teal')
plt.xlabel('Absolute Coefficient Value', fontweight='bold')
plt.title('Feature Importance', fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 4. Distribution of Predictions vs Actual
plt.subplot(2, 3, 4)
plt.hist(y_test, bins=30, alpha=0.6, label='Actual', color='blue', edgecolor='black')
plt.hist(y_test_pred, bins=30, alpha=0.6, label='Predicted', color='red', edgecolor='black')
plt.xlabel('Price ($)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Distribution: Actual vs Predicted', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Price vs Size (with predictions)
plt.subplot(2, 3, 5)
test_indices = X_test.index
plt.scatter(X.loc[test_indices, 'size_sqft'], y_test, alpha=0.6, 
           label='Actual', edgecolors='k', linewidths=0.5)
plt.scatter(X.loc[test_indices, 'size_sqft'], y_test_pred, alpha=0.6, 
           label='Predicted', marker='^', edgecolors='k', linewidths=0.5)
plt.xlabel('Size (sqft)', fontweight='bold')
plt.ylabel('Price ($)', fontweight='bold')
plt.title('Price vs Size', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Error Distribution
plt.subplot(2, 3, 6)
plt.hist(residuals, bins=30, color='coral', edgecolor='black')
plt.xlabel('Prediction Error ($)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title(f'Error Distribution\nMean Error: ${residuals.mean():,.0f}', fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_prediction.png', dpi=300, bbox_inches='tight')
print("\n\nVisualizations saved as 'house_price_prediction.png'")
plt.show()

# SAMPLE PREDICTIONS
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample_houses = X_test.head(5)
sample_predictions = model.predict(scaler.transform(sample_houses))
sample_actual = y_test.head(5)

print("\nPredictions for 5 sample houses:\n")
for i, (idx, house) in enumerate(sample_houses.iterrows()):
    print(f"House {i+1}:")
    print(f"  Rooms: {house['num_rooms']}, Size: {house['size_sqft']} sqft, " +
          f"Location Score: {house['location_score']:.1f}")
    print(f"  Actual Price: ${sample_actual.iloc[i]:,.0f}")
    print(f"  Predicted Price: ${sample_predictions[i]:,.0f}")
    print(f"  Error: ${abs(sample_actual.iloc[i] - sample_predictions[i]):,.0f}\n")

print("=" * 70)