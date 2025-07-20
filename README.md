# AMEX Dataset Analysis Project

This repository contains a comprehensive analysis of the American Express (AMEX) dataset for the Campus Challenge 25. The project includes data exploration, preprocessing, machine learning pipeline development, and model training.

## 📁 Project Structure

```
├── README.md                           # This file
├── .gitignore                          # Git ignore file
├── quick_analysis.py                   # Quick data analysis script
├── explore_parquet.py                  # Parquet file exploration utility
├── test_ml_pipeline.py                 # ML pipeline testing script
├── basic_ml_pipeline.ipynb            # Jupyter notebook for ML pipeline
├── amex_data_exploration.ipynb        # Data exploration notebook
└── venv/                              # Virtual environment (not tracked)
```

## 🎯 Project Overview

This project analyzes the AMEX dataset to understand customer behavior patterns and develop predictive models. The analysis includes:

- **Data Exploration**: Understanding the structure and characteristics of the dataset
- **Data Preprocessing**: Cleaning and preparing data for machine learning
- **Feature Engineering**: Creating relevant features for model training
- **Model Development**: Building and evaluating machine learning models
- **Pipeline Testing**: Ensuring reproducibility and reliability

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd amex-dataset-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

### Dataset Setup

⚠️ **Important**: The dataset files are not included in this repository due to their large size. You'll need to:

1. Download the AMEX dataset files:
   - `train_data.parquet`
   - `test_data.parquet`
   - `add_trans.parquet`
   - `add_event.parquet`
   - `offer_metadata.parquet`
   - `data_dictionary.csv`
   - `685404e30cfdb_submission_template.csv`
   - `68524025db373_Campus_Challenge25_Amex_Offer.pdf`

2. Place these files in the root directory of the project

## 📊 Files Description

### Analysis Scripts

- **`quick_analysis.py`**: Performs initial data exploration and generates summary statistics
- **`explore_parquet.py`**: Utility script for exploring parquet file structure and contents
- **`test_ml_pipeline.py`**: Tests the machine learning pipeline with sample data

### Jupyter Notebooks

- **`basic_ml_pipeline.ipynb`**: Complete machine learning pipeline development
- **`amex_data_exploration.ipynb`**: Comprehensive data exploration and visualization

### Generated Files (Not Tracked)

- **`data_preprocessor.pkl`**: Saved data preprocessing pipeline
- **`best_model_gradient_boosting.pkl`**: Trained gradient boosting model
- **`venv/`**: Virtual environment directory

## 🔧 Usage

### Quick Analysis
```bash
python quick_analysis.py
```

### Explore Parquet Files
```bash
python explore_parquet.py
```

### Test ML Pipeline
```bash
python test_ml_pipeline.py
```

### Jupyter Notebooks
```bash
jupyter notebook
```
Then open either `basic_ml_pipeline.ipynb` or `amex_data_exploration.ipynb`

## 📈 Key Features

- **Data Preprocessing**: Automated cleaning and feature engineering
- **Model Training**: Gradient boosting and other ML algorithms
- **Pipeline Persistence**: Save and load trained models and preprocessors
- **Comprehensive Analysis**: Statistical summaries and visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is for educational and research purposes. Please refer to the AMEX Campus Challenge terms and conditions for dataset usage.

## 📞 Contact

For questions or issues related to this analysis, please open an issue in the repository.

---

**Note**: This repository contains only the analysis code and scripts. The actual dataset files are not included due to size constraints and licensing requirements. Please ensure you have proper access to the AMEX dataset before running the analysis scripts. 