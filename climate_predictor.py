"""
ClimateChangePredictor - AI for Carbon Emission Forecasting
SDG 13: Climate Action - Take urgent action to combat climate change
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

class ClimatePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_importance = None
        
    def generate_synthetic_data(self):
        """Generate realistic climate and emission data"""
        np.random.seed(42)
        years = np.arange(2000, 2023)
        
        data = {
            'year': years,
            'co2_emissions': 30000 + years * 80 + np.random.normal(0, 500, len(years)),
            'gdp_per_capita': 20000 + years * 300 + np.random.normal(0, 1000, len(years)),
            'population': 1e6 + years * 25000 + np.random.normal(0, 50000, len(years)),
            'renewable_energy': 10 + years * 0.8 + np.random.normal(0, 2, len(years)),
            'forest_area': 30 - years * 0.1 + np.random.normal(0, 0.5, len(years)),
            'temperature_anomaly': 0.2 + years * 0.03 + np.random.normal(0, 0.1, len(years))
        }
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        """Prepare features and target variable"""
        # Features: economic and environmental indicators
        features = ['gdp_per_capita', 'population', 'renewable_energy', 'forest_area', 'temperature_anomaly']
        
        # Target: CO2 emissions (what we want to predict)
        X = df[features]
        y = df['co2_emissions']
        
        return X, y
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test, y_test, y_pred, mae, r2
    
    def predict_future(self, X, future_years=5):
        """Predict emissions for future years"""
        # Create future scenario (modify these based on different policies)
        future_data = []
        current_values = X.iloc[-1:].copy()
        
        for i in range(1, future_years + 1):
            future_row = current_values.copy()
            # Project trends - you can modify these for different scenarios
            future_row['gdp_per_capita'] *= 1.02  # 2% GDP growth
            future_row['population'] *= 1.01       # 1% population growth
            future_row['renewable_energy'] += 1.5  # Increased renewable adoption
            future_row['forest_area'] -= 0.05      # Slight deforestation
            future_row['temperature_anomaly'] += 0.03  # Warming trend
            
            future_data.append(future_row.copy())
        
        future_df = pd.concat(future_data, ignore_index=True)
        future_emissions = self.model.predict(future_df)
        
        return future_df, future_emissions
    
    def visualize_results(self, df, X_test, y_test, y_pred, future_emissions):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.7)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Emissions')
        axes[0, 0].set_ylabel('Predicted Emissions')
        axes[0, 0].set_title('Actual vs Predicted CO2 Emissions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature Importance
        sns.barplot(data=self.feature_importance, x='importance', y='feature', ax=axes[0, 1])
        axes[0, 1].set_title('Feature Importance in Emission Prediction')
        axes[0, 1].set_xlabel('Importance Score')
        
        # Plot 3: Historical Trends
        axes[1, 0].plot(df['year'], df['co2_emissions'], marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('CO2 Emissions (kt)')
        axes[1, 0].set_title('Historical CO2 Emissions Trend')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Future Projections
        future_years = list(range(2023, 2023 + len(future_emissions)))
        axes[1, 1].plot(df['year'], df['co2_emissions'], label='Historical', marker='o')
        axes[1, 1].plot(future_years, future_emissions, label='Predicted', marker='s', linestyle='--')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('CO2 Emissions (kt)')
        axes[1, 1].set_title('Future CO2 Emissions Projection')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/results_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🌍 ClimateChangePredictor - AI for SDG 13: Climate Action")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ClimatePredictor()
    
    # Generate and display data
    print("📊 Generating climate and emissions data...")
    df = predictor.generate_synthetic_data()
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Prepare data and train model
    print("\n🤖 Training Machine Learning Model...")
    X, y = predictor.prepare_data(df)
    X_test, y_test, y_pred, mae, r2 = predictor.train_model(X, y)
    
    print(f"✅ Model Performance:")
    print(f"   Mean Absolute Error: {mae:.2f}")
    print(f"   R² Score: {r2:.2f}")
    
    # Feature importance
    print(f"\n🔍 Top Features Affecting Emissions:")
    for _, row in predictor.feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Future predictions
    print(f"\n🔮 Predicting Future Emissions (2023-2027)...")
    future_df, future_emissions = predictor.predict_future(X)
    
    for i, emission in enumerate(future_emissions, 2023):
        print(f"   {i}: {emission:.0f} kt CO2")
    
    # Visualizations
    print(f"\n📈 Generating visualizations...")
    predictor.visualize_results(df, X_test, y_test, y_pred, future_emissions)
    
    # Ethical considerations
    print(f"\n⚖️ ETHICAL CONSIDERATIONS:")
    print("   • Data Bias: Synthetic data may not capture regional variations")
    print("   • Fairness: Ensure predictions don't disproportionately impact developing nations")
    print("   • Transparency: Model decisions should be explainable to policymakers")
    print("   • Actionability: Predictions should lead to concrete climate policies")

if __name__ == "__main__":
    main()