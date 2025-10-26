
# AIT-511 Machine Learning Project — Checkpoint 1
## Medical Equipments Cost Prediction Challenge

## Team Details

### Team Name: Hidden Outliers
### Team members:
- Shreeya H (IMT2022535)
- Santhosh Kiran (IMT2023065)
- Sriram Srikanth (IMT2023115)

## Repository folder structure
## Folder Structure
```
.
├── images
├── main.py
├── Preprocesor.py
├── README.md
├── requirements.txt
├── test.csv
└── train.csv
└── project_report.pdf
```

## What each file does

- `main.py`
	- Loads `train.csv` and `test.csv`.
	- Calls preprocessing from `Preprocesor.py`.
	- Trains the chosen model(s), evaluates results, and writes outputs/metrics.

- `Preprocesor.py`
	- Implements preprocessing functions used by `main.py`.
	- It preprocesses the train,test dataset by imputing NaN values, engineering features, fixing outlying, skewed data, encoding non-numeric features and scaling the numeric features.

- `train.csv` / `test.csv`
    - These are the labeled training dataset used to train the model and test dataset used for final predictions.

- `images/`
	- Stores images produced during EDA, model results used in the report.

-  `requirements.txt`
    - Contains the list of necessary python libraries to be installed to run the code

- `project_report.pdf`
    - The project report contains all information regarding the problem statement,dataset, preprocessing steps, various models experiments and best predictions.


## Run Locally

Clone the project

```bash
  git clone https://github.com/Humanoid2005/ML_Project_Checkpoint1
```

Go to the project directory

```bash
  cd ML_Project_Checkpoint1
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the main program and enter the model you want to use

```bash
  py main.py
```


