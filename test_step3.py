from ml_training.utils.data_loader import load_churn_features

df = load_churn_features()
print(f"Shape: {df.shape}")
print(f"Churn rate: {df['label'].mean():.1%}")
print(f"Number of features: {len(df.columns)}")
print(f"Features: {list(df.columns)}")