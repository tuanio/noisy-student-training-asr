# Introduction

Automatic Pronunciation Error Detection (APED).

An APED system will first provide a predefined text (and, if necessary, a pre-existing voiceover for learners to listen to for reference). The learner's task is very simple: try to read this passage as correctly as possible. For example, a learner wants to learn how to pronounce the word “apple” (its phoneme sequence is “æ p l”), but the learner may mispronounce it as “ə p l”. In this case, we define the string "æ p l" as the standard pronunciation string and the string "ə p l" as the reader's string. The APED system will accurately predict where the user reads the wrong word "apple" in a specific position, thereby giving feedback to the learner so that the learner can promptly correct the mistake, gradually, the learner will improve your pronunciation.

# Method

# Result

conda create -n dev -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.9 pytorch torchaudio cudatoolkit pandas numpy 

pip install fairseq transformers torchsummary datasets evaluate torch-summary jiwer wandb matplotlib

wandb login 2cfd4b5c7e2828d412e5f871efea3a4c582efe18
