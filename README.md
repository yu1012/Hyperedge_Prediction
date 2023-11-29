# MHP
This is a repository of `MHP`.

## Data
Make a `hypergraph_data` directory outside the current directory, and copy and unzip data.zip there.

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
python eval.py --dataset [data_name] --split [data_split] --gpu [gpu_id]
```
