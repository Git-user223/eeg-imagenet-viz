# EEG-ImageNet-Viz


<img width="600" alt="image" src="/assets/dataset_overview_diagram.png">

### Interactive visualizations and evaluation tools for **EEG-ImageNet**–style experiments. This repo helps you:

- Preprocess EEG data
- Featurize EEG Data
- Train 4 Models on the featurized data (LR, SVM, RF, MLP)
- Evaluate the performance of the classification models
- Plot the results
- Create an interactive dashboard of the results and data exploration

### Project Structure: 
```
project-root/
├── app/ # Contains Streamlit runnable app python file
├── assets/ # Static Images used in the dashboard 
|
├── data/ # Data from all steps of the preprocessing/training/visualization steps
│ ├── raw/ # Original .pth/.npz EEG datasets (too big to upload)
│ ├── interim/ # Preprocessed EEG saved as .pkl files (too big to upload)
│ ├── processed/ # Feature tables saved as .parquet (1 example included)
│ ├── results/ # Model outputs: predictions, summary.csv, trained .joblib models (all included)
│ └── plots/ # Generated html visualizations (accuracy, confusion matrices, top-k plots) 
|
├── plots/ # Plots included in the dashboard 
├── results/ # All evalaution results included in the dashboard
├── src/ # Preprocessing scripts used to load and clean the data as well as train and evaluate models
├── environment.yaml # Requirements file for environment reproduceability.
│
└── README.md # Project documentation
```
---