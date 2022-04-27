# Generating Repetitions with Appropriate Repeated Words
Code repository for Generating Repetitions with Appropriate Repeated Words of NAACL 2022 [URL]

## Usage
### Setup
```bash
git clone https://github.com/titech-nlp/repetition-generation

cd repetition-generation

poetry install
```
Download repetition dataset from [URL](https://1drv.ms/u/s!AndxLE_vhGP2hk6jZRRC922vPGWt?e=LzTgBl)  
(License of dataset: CC BY-NC 4.0)

Set the dataset to `data/`

### Preprocess and calculate repeat score  
```bash
export PYTHONPATH='./'
poetry run python scripts/preprocess.py
```

### Training
```bash
poetry run python scripts/train.py
```

Or you can download our trained model from [URL](https://1drv.ms/u/s!AndxLE_vhGP2hk15V7E0Io3jdr7f?e=AmxeLz) and set it to `models/`

### Test
```bash
poetry run python scripts/test.py
```


## Citation
Coming soon!