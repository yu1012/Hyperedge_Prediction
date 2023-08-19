# Hyperedge_Prediction

[Project Description: Provide a brief overview of your project and its main goals. Mention what hyperedge prediction is and why it's important.]

## Data
Make a `hypergrad_data` directory outside the current directory, and copy files from the 
https://drive.google.com/drive/folders/1yASDHE-9tD0byFgul22azmN9vZw_Jw_X?usp=sharing

## Installation
We implemented MHP with python 3.9. To install packages, run the shell code below.
```bash
pip install -r requirements.txt
```

## Usage
To train MHP,
```bash
python train.py --dataset [data_name] --split [data_split] --gpu [gpu_id] --epochs [num_epochs]
```

To evaluate MHP,
```
python eval.py --dataset [data_name] --split [data_split] --gpu [gpu_id] --epochs [num_epochs]
```
