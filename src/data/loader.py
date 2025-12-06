import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

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
    
    def load_excel(self, filename, sheet_name=0):
        """Load Excel file from raw data directory."""
        filepath = self.raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_excel(filepath, sheet_name=sheet_name)
    
    def save_processed(self, df, filename):
        """Save processed data to processed directory."""
        filepath = self.processed_path / filename
        self.processed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath
    
    def get_sample_medical_data(self, n_samples=1000):
        """Generate sample medical data for testing."""
        np.random.seed(42)
        
        data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'blood_pressure_sys': np.random.normal(120, 15, n_samples),
            'blood_pressure_dia': np.random.normal(80, 10, n_samples),
            'cholesterol': np.random.choice(['normal', 'high', 'very_high'], n_samples),
            'glucose': np.random.normal(100, 20, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_disease': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def load_dicom_metadata(self, dicom_dir):
        """Load DICOM file metadata (for medical imaging)."""
        import pydicom
        from pathlib import Path
        
        dicom_files = list(Path(dicom_dir).glob('*.dcm'))
        metadata_list = []
        
        for file_path in dicom_files[:10]:  # Load first 10 files
            try:
                ds = pydicom.dcmread(str(file_path))
                metadata = {
                    'filename': file_path.name,
                    'patient_id': getattr(ds, 'PatientID', 'N/A'),
                    'modality': getattr(ds, 'Modality', 'N/A'),
                    'study_date': getattr(ds, 'StudyDate', 'N/A'),
                    'rows': getattr(ds, 'Rows', 0),
                    'columns': getattr(ds, 'Columns', 0),
                    'bits_allocated': getattr(ds, 'BitsAllocated', 0)
                }
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return pd.DataFrame(metadata_list)
    
    def check_data_quality(self, df):
        """Check data quality metrics."""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return quality_report

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    sample_data = loader.get_sample_medical_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {sample_data.columns.tolist()}")
    print(f"First 5 rows:\n{sample_data.head()}")
    
    # Check data quality
    quality = loader.check_data_quality(sample_data)
    print("\nData Quality Report:")
    for key, value in quality.items():
        print(f"{key}: {value}")