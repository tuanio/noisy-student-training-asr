```
**Pseudo Code**
function LCS(s1, s2):
  n = len(s1)
  m = len(s2)
  L = empty_array(0..n, 0..m)
  for i := 1 to :n
    for j := 1 to m:
      elif s1[i - 1] == s2[j - 1]:
        L[i][j] := add character s1[i - 1] to array L[i - 1][j - 1]
      else:
        L[i][j] := max(L[i - 1][j], L[i][j - 1], key=len)

  return L[n][m]

s1 = "ɔ l ɪ z s ɛ d w ɪ ð aʊ t ə w ə d" of length n
s2 = "ɔ l w ɪ z s ɛ d w ɪ ð aʊ t ə w ə d" of length m

lcs = LCS(s1, s2)

print("Longest Common Phoneme Sequence:", lcs)
***Output
Longest Common Phoneme Sequence: ɔ l ɪ z s ɛ d w ɪ ð aʊ t ə w ə d
***
```
