# Dataset guide

## Folder layout

The training code expects one folder per label and WAV files inside each folder:

```text
data/raw/
├── angry/
├── happy/
├── neutral/
└── sad/
```

Folder names become the target labels. Use the labels provided by your chosen dataset and avoid mixing inconsistent naming schemes.

## Choosing data

Pick a public speech-emotion dataset only after reading its terms of use, license, consent conditions, and redistribution restrictions. Some datasets allow research use but prohibit redistribution or commercial use. Do not upload the raw dataset to this GitHub repository unless its license explicitly permits that.

For a first experiment, use a small subset locally and document:
- Dataset name and official source
- License / permitted use
- Number of clips per class
- Sample rate and audio format
- Whether recordings are acted or naturally occurring

## Evaluation caution

Randomly splitting audio clips can overestimate performance if the same speaker appears in both train and test sets. For a more realistic result, split by speaker where speaker IDs are available. The current training script uses a stratified random split for a simple baseline; do not present that score as proof of real-world accuracy.
