# Introduction

## Noisy Student Training

Noisy Student Training is a semi-supervised learning approach. It extends the idea of self-training and distillation with the use of equal-or-larger student models and noise added to the student during learning.

It has three main steps:

- 1. Train a teacher model on labeled speech.
- 2. Use the teacher to generate pseudo labels on unlabeled speech.
- 3. Train a student model on the combination of labeled speech and pseudo labeled speech.

The algorithm is iterated a few times by treating the student as a teacher to relabel the unlabeled data and training a new student.

Noisy Student Training seeks to improve on self-training and distillation in two ways. First, it makes the student larger than, or at least equal to, the teacher so the student can better learn from a larger dataset. Second, it adds noise to the student so the noised student is forced to learn harder from the pseudo labels. In Automatic Speech Recognition (ASR), to noise the student, it uses input noise such as speed Perturbation, Multi-style Training, Spec Augmentation data augmentation, and/or model noise such as dropout and stochastic depth during training.

![nst](https://user-images.githubusercontent.com/30165828/213200362-0d071be1-2922-4a55-a046-11b848a130c2.png)

## About

This GitHub is my source code when I did my graduate thesis on Automatic Pronunciation Error Detection (APED) based on ASR. With just a little tweaking of the input, this source code can be used well for speech recognition problems.

# How to use

## Structure

## Dataset

## Model config

# Automatic Pronunciation Error Detection (APED).

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

Dataset can get from here [tuannguyenvananh/libri-phone](https://www.kaggle.com/datasets/tuannguyenvananh/libri-phone).

## Language Model
- Language Model is 3-gram Witten-Bell LM, train on **label** phoneme corpus (too small)

| Corpus | No. Sentences | Perplexity |
| --- | --- | --- |
| Training: $80$% | 8906 | 10.17 | 
| Testing: $20$% | 2241 | 10.47 |

## Results

### Phoneme Recognition

- Metric: Phoneme Error Rate (%)

| Model | Greedy Decode | Beam Search (with LM) Decode |
| --- | --- | --- |
| Teacher | $17.13$ | $22.31$ |
| Student | **$12.66$** | $25.45$ |

The result show that:
- The language model is not suitable for this problem because of the small amount of text data.
- Student's PER is reduced by 26% compared to that of teacher, successfully implementing supervised learning technique for the problem of phonemic sequence recognition.

#### Some example on Phoneme Sequence Prediction

| - | Ground Truth | Predict | PER (%) |
| --- | --- | --- | --- |
| 1 | w aɪ ə t ʌ ŋ ɪ m p ɹ ɛ s d w ɪ ð h ʌ n i f ɹ ʌ m ɛ v ɹ i w ɪ n d | w aɪ ə t **t ɑː** ŋ ɪ m p ɹ ɛ s t w ɪ ð h ʌ n i f ɹ ʌ m ɛ v ɹ i w ɪ n d ɪ | $12.5$ | 
| 2 | w aɪ ə n ɪ ɹ ə w ə l p uː l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | w aɪ ə n ɪ ɹ ə w ə l p uː l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | $0$ | 
| 3 | ɔ l ɪ z s ɛ d w ɪ ð aʊ t ə w ə d | ɔ l **w** ɪ z s ɛ d w ɪ ð aʊ t ə w ə d | $6.25$ | 
| 4 | aɪ s ɪ t b ə n i θ **ð** aɪ l ʊ k s æ z t ʃ ɪ l d ɹ ə n d uː ɪ n ð ə n uː n s ʌ n w ɪ ð s oʊ l z ð æ t **t** ɹ ɛ m b l θ ɹ uː ð ɛ ɹ h æ p i aɪ l ɪ **d** z f ɹ ʌ m ə n ʌ n ə v ə d j ɛ t p ɹ ɑː d ɪ ɡ l ɪ n w ɚ d d ʒ ɔɪ | aɪ s ɪ t b ə n i θ aɪ l ʊ k s æ z t ʃ ɪ l d ɹ ə n d uː ɪ n ð ə n uː n s ʌ n w ɪ ð s oʊ l z ð æ t ɹ ɛ m b l θ ɹ uː ð ɛ ɹ h æ p i aɪ **ə** l ɪ z f ɹ ʌ m ə n ʌ n **ʌ** v ə d j ɛ t p ɹ ɑː **n** ɪ **k** l ɪ n w ɚ d d ʒ ɔɪ ə **ɪ ɪ** | $10.2$ | 

### Error Detection

#### Setup

| Information | Value |
| --- | --- |
| Grapheme | why an ear a whirlpool fierce to draw creations in | 
| IPA Phoneme | w aɪ ə n ɪ ɹ ə w ə l p uː l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n |
| Phoneme sequence length | 32 | 
| Path to LibriSpeech test-clean | test-clean/908/157963/908-157963-0030.flac |

To perform error checking, change the letter form of the sample sentence to another word in turn, then recreate the IPA phonetic form, and then use the student model to predict and check errors using the longest common subsequence (LCS) algorithm.

Generate IPA with [bootphon/phonemizer](https://github.com/bootphon/phonemizer) and generate voice using [Nvidia FastPitch's text-to-speech API](https://huggingface.co/nvidia/tts_en_fastpitch)

| Example Name | Grapheme | Original Phoneme Sequence | Phoneme Sequence Length | 
| --- | --- | --- | --- |
| `error_1` | why an ear a w**eir** pool fierce to draw creations in | w aɪ ə n **ɪ ɹ** ə w ɪ ɹ p uː l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | 32 |
| `error_2` | why an ear a whirlpool **fear** to draw creations in | w aɪ ə n ɪ ɹ ə w ə l p uː l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | 31 |
| `error_3` | why an ear a whirl **pole** fierce to draw creations in | w aɪ ə n ɪ ɹ ə w ə l p **oʊ** l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | 32 |

Above Dataset can get from here [tuannguyenvananh/aped-sample](https://www.kaggle.com/datasets/tuannguyenvananh/aped-sample).

#### Error Detection Result

Using the LCS algorithm to match the longest sequence between the real sample and the predicted sample, unmatched phonemes will be considered as pronunciation errors.

| Example Name | Predicted Phoneme Sequence | Phoneme that unmatch by LCS | Predicted Sequence Length | PER (%) | Result | 
 | --- | --- | --- | --- | --- | --- |
 | `error_1` | w aɪ ə n ɪ ɹ ə w ɪ ɹ p uː l f **ɪ ɹ** s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | ɪ, ɹ | 32 | $6.25$ | <span style="color: green;">Correct</span> |
  | `error_2` | w aɪ ə n ɪ ɹ ə w ə **k** l f ɪ ɹ t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | k | 29 | $12.5$ | <span style="color: red;">Wrong</span> |
   | `error_3` | w aɪ ə n ɪ ɹ ə w ə l p **oʊ** l f ɪ ɹ s t ə d ɹ ɔ k ɹ i eɪ ʃ ə n z ɪ n | oʊ | 32 | $3.125$ | <span style="color: green;">Correct</span> |

# Installation

## Create Conda environment 

`conda create -n dev -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.9 pytorch torchaudio cudatoolkit pandas numpy`

## Install requirement packages

`pip install fairseq transformers torchsummary datasets evaluate torch-summary jiwer wandb matplotlib`

## Login Wandb 

`wandb login <token>`

# References

- **[1]** Yu. et el. "Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition". DOI: [doi.org/10.48550/arXiv.2010.10504](https://doi.org/10.48550/arXiv.2010.10504).

# Pretrained Checkpoint

You guys can get checkpoint file from here [tuannguyenvananh/nst-pretrained-model](https://www.kaggle.com/datasets/tuannguyenvananh/nst-pretrained-model).

# Report

The final report can get from here (Vietnamese version): [graduation-thesis_v3.pdf](https://github.com/tuanio/noisy-student-training-aped/files/10369639/graduation-thesis_v3.pdf).

# Things to do

- [ ] Restructure code.
- [ ] Publish pretrained for Teacher and Student.

<!-- wandb login 2cfd4b5c7e2828d412e5f871efea3a4c582efe18 -->
