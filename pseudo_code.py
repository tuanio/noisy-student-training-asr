import numpy as np

s1 = "h i h oʊ p t ð ɛ ɹ w ʊ d b i s t uː f ɔː ɹ d ɪ n ɚ t ə n ɪ p s æ n d k æ ɹ ə t s æ n d b ɹ uː z d p ə t eɪ ɾ oʊ z æ n d f æ t m ʌ n p i s ə z t ə b i l æ d l d aʊ t ɪ n θ ɪ k p ɛ p ɚ d f l aʊ ɚ f æ n d s ɔ s".split()
s2 = "h j h b æ ʌ ɚ f uː v n ɾ ɛ j d ʌ ɑː ɑ ɔ uː ɾ ð ŋ ʊɹ ʌ z ŋ ð æ d ɔɪ ŋ ɾ ʒ ɔɪ uː ð ʌ d ɔɪ ŋ ɾ ɛ uː ɑː ə ɾ ɛ z ʌ k ɾ h b ə ɔɪ ŋ ɾ ɑ ɔɪ ʌ eɪ oʊ ʌ z ŋ æ j d z ə ʌ z ɛ j aɪ k ɾ aɪ ɾ ɹ ʌ ð ŋ ɡ ð ʒ æ f æ ʊɹ ɾ ɑ aɪ ɹ ʊɹ ɑ ɔɪ z ŋ d m d".split()

n = len(s1)
m = len(s2)

L = np.zeros((n + 1, m + 1)).tolist()

for i in range(0, n + 1):
    for j in range(0, m + 1):
        if not i or not j:
            L[i][j] = []
        elif s1[i - 1] == s2[j - 1]:
            L[i][j] = L[i - 1][j - 1] + [s1[i - 1]]
        else:
            L[i][j] = max(L[i - 1][j], L[i][j - 1], key=len)

print(L[n][m], len(L[n][m]))
