# Introduction

Automatic Pronunciation Error Detection (APED).

An APED system will first provide a predefined text (and, if necessary, a pre-existing voiceover for learners to listen to for reference). The learner's task is very simple: try to read this passage as correctly as possible. For example, a learner wants to learn how to pronounce the word “apple” (its phoneme sequence is “æ p l”), but the learner may mispronounce it as “ə p l”. In this case, we define the string "æ p l" as the standard pronunciation string and the string "ə p l" as the reader's string. The APED system will accurately predict where the user reads the wrong word "apple" in a specific position, thereby giving feedback to the learner so that the learner can promptly correct the mistake, gradually, the learner will improve your pronunciation.

![aped_asr_based](https://user-images.githubusercontent.com/30165828/211220506-ff92e3e7-e42b-4146-902f-98c94a71569b.png)

# Method

- 1. Training Conformer using Pre-training Wav2Vec2.0 Framework combined with Self-training technique Noisy Student Training for Phoneme Recognition problem.

- 2. Find the longest common subsequence between the ground truth phoneme sequence and model-predicted speaker phoneme sequence to detect miss-pronunciation.

![project_structure](https://user-images.githubusercontent.com/30165828/211220527-67ff9432-30a8-45df-bac2-57072624c1eb.png)

# Experiment

## Dataset

### Type of Label

Label is phoneme sequence, extracted from [bootphon/phonemizer](https://github.com/bootphon/phonemizer).

### Speech Dataset

#### Training
- **Pretraining wav2vec2.0 Conformer**: Using Pretrained of [facebook/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20). `Wav2Vec 2.0 Large conformer - rel_pos (LV-60)`.
- **Unlabel**: LibriSpeech 360 hours remove label.
- **Label**: Libri-Light 10h + LibriSpeech dev-clean, dev-other, dev-other. Total 25 hours.

#### Testing
- **Label**: LibriSpeech test-clean 5.4 hours.

## Language Model
- Language Model is 3-gram Witten-Bell LM, train on **label** phoneme corpus (too small)

| Corpus | No. Sentences | Perplexity |
| --- | --- | --- |
| Training: $80$% | 8906 | 10.17 | 
| Testing: $20$% | 2241 | 10.47 |

## Results

### Phoneme Recognition

- Metric: Word Error Rate (%)

| Model | Greedy Decode | Beam Search (with LM) Decode |
| --- | --- | --- |
| Teacher | $17.13$ | $22.31$ |
| Student | **$12.66$** | $25.45$ |

### Error Detection

# Installation

## Create Conda environment 

`conda create -n dev -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.9 pytorch torchaudio cudatoolkit pandas numpy`

## Install requirement packages

`pip install fairseq transformers torchsummary datasets evaluate torch-summary jiwer wandb matplotlib`

## Login Wandb 

`wandb login <token>`

# References

- **[1]** Yu. et el. "Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition". DOI: [doi.org/10.48550/arXiv.2010.10504](https://doi.org/10.48550/arXiv.2010.10504).

# Things to do

- [ ] Restructure code.
- [ ] Publish pretrained for Teacher and Student.

<!-- wandb login 2cfd4b5c7e2828d412e5f871efea3a4c582efe18 -->
