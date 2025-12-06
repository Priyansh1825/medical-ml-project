import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class DataLoader:
    """Medical data loader for ML projects."""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['data']['raw_data_path'])
        self.processed_path = Path(self.config['data']['processed_data_path'])
        
    def load_csv(self, filename):
        """Load CSV file from raw data directory."""
        filepath = self.raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)
    
    def save_processed(self, df, filename):
        """Save processed data to processed directory."""
        filepath = self.processed_path / filename
        self.processed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath
    
    def get_sample_medical_data(self):
        """Generate sample medical data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'blood_pressure_sys': np.random.normal(120, 15, n_samples),
            'blood_pressure_dia': np.random.normal(80, 10, n_samples),
            'cholesterol': np.random.choice(['normal', 'high', 'very_high'], n_samples),
            'glucose': np.random.normal(100, 20, n_samples),
            'has_disease': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    sample_data = loader.get_sample_medical_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {sample_data.columns.tolist()}")
    print(f"First 5 rows:\n{sample_data.head()}")
