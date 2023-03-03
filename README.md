# Generating Repetitions with Appropriate Repeated Words
Code repository for Generating Repetitions with Appropriate Repeated Words of NAACL 2022
https://aclanthology.org/2022.naacl-main.62/

## Usage
### Setup
```bash
git clone https://github.com/titech-nlp/repetition-generation

cd repetition-generation

poetry install
```
Download repetition dataset from [URL](https://1drv.ms/u/s!AndxLE_vhGP2hk6jZRRC922vPGWt?e=LzTgBl)
(License of dataset: CC BY-NC 4.0)

Unzip and set the dataset to `data/repetition/`

### Preprocess and calculate repeat score
```bash
export PYTHONPATH='./'
poetry run python scripts/preprocess.py
```
This process trains a language model to prepare repeat score.

### Training
```bash
poetry run python scripts/train.py
```

Or you can download our trained model from [URL](https://1drv.ms/u/s!AndxLE_vhGP2hk15V7E0Io3jdr7f?e=AmxeLz) and unzip and set it to `models/`

### Test
```bash
poetry run python scripts/test.py
```


## Citation
```
@inproceedings{kawamoto-etal-2022-generating,
    title = "Generating Repetitions with Appropriate Repeated Words",
    author = "Kawamoto, Toshiki  and
      Kamigaito, Hidetaka  and
      Funakoshi, Kotaro  and
      Okumura, Manabu",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.62",
    pages = "852--859",
    abstract = "A repetition is a response that repeats words in the previous speaker{'}s utterance in a dialogue. Repetitions are essential in communication to build trust with others, as investigated in linguistic studies. In this work, we focus on repetition generation. To the best of our knowledge, this is the first neural approach to address repetition generation. We propose Weighted Label Smoothing, a smoothing method for explicitly learning which words to repeat during fine-tuning, and a repetition scoring method that can output more appropriate repetitions during decoding. We conducted automatic and human evaluations involving applying these methods to the pre-trained language model T5 for generating repetitions. The experimental results indicate that our methods outperformed baselines in both evaluations.",
}
```