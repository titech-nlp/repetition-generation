# Generating Repetitions with Appropriate Repeated Words
Code repository for Generating Repetitions with Appropriate Repeated Words of NAACL 2022 [URL]

## Usage
### Setup
```bash
git clone https://github.com/titech-nlp/repetition-generation

cd repetition-generation

poetry install
```
Download repetition dataset from URL  
(License of dataset: CC BY-NC 4.0)

Set the dataset to `data/`

### Preprocess and calculate repeat score  
```bash
export PYTHONPATH='./'
poetry run python preprocess.py
```

### Training
```bash
poetry run python train.py
```

Or you can download trained model from URL and set it to `models/`

Test
```bash
poetry run python test.py
```


## Citation
coming soon...