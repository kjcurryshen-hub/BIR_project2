"""Porter 詞幹還原演算法 (Porter, 1980)。

公開領域演算法，依論文 "An algorithm for suffix stripping"
(Program, 14(3): 130-137) 之五個步驟實作，供本系統索引前處理使用。
"""

VOWELS = "aeiou"


class PorterStemmer:
    def __init__(self):
        self.b = ""   # 目前處理的字串
        self.k = 0    # 字尾索引 (含)
        self.j = 0    # 一般用途的位移

    # ---- 基本判斷 -------------------------------------------------
    def _cons(self, i):
        """b[i] 是否為子音。"""
        ch = self.b[i]
        if ch in VOWELS:
            return False
        if ch == "y":
            return True if i == 0 else not self._cons(i - 1)
        return True

    def _m(self):
        """計算 b[0:j+1] 的 measure，即 (VC) 重複出現的次數。"""
        n = 0
        i = 0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self):
        return any(not self._cons(i) for i in range(self.j + 1))

    def _double_cons(self, j):
        if j < 1 or self.b[j] != self.b[j - 1]:
            return False
        return self._cons(j)

    def _cvc(self, i):
        """b[i-2:i+1] 為 子音-母音-子音，且末字元不是 w/x/y。"""
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s):
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        self.b = self.b[:self.j + 1] + s
        self.k = self.j + len(s)

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    # ---- 五個步驟 -------------------------------------------------
    def _step1ab(self):
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._double_cons(self.k):
                if self.b[self.k] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        if self._ends("y") and self._vowel_in_stem():
            self.b = self.b[:self.k] + "i"

    def _step2(self):
        ch = self.b[self.k - 1] if self.k > 0 else ""
        table = {
            "a": [("ational", "ate"), ("tional", "tion")],
            "c": [("enci", "ence"), ("anci", "ance")],
            "e": [("izer", "ize")],
            "l": [("bli", "ble"), ("alli", "al"), ("entli", "ent"),
                  ("eli", "e"), ("ousli", "ous")],
            "o": [("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            "s": [("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
                  ("ousness", "ous")],
            "t": [("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            "g": [("logi", "log")],
        }
        for suffix, repl in table.get(ch, []):
            if self._ends(suffix):
                self._r(repl)
                return

    def _step3(self):
        ch = self.b[self.k]
        table = {
            "e": [("icate", "ic"), ("ative", ""), ("alize", "al")],
            "i": [("iciti", "ic")],
            "l": [("ical", "ic"), ("ful", "")],
            "s": [("ness", "")],
        }
        for suffix, repl in table.get(ch, []):
            if self._ends(suffix):
                self._r(repl)
                return

    def _step4(self):
        ch = self.b[self.k - 1] if self.k > 0 else ""
        table = {
            "a": ["al"], "c": ["ance", "ence"], "e": ["er"], "i": ["ic"],
            "l": ["able", "ible"], "n": ["ant", "ement", "ment", "ent"],
            "o": ["ion", "ou"], "s": ["ism"], "t": ["ate", "iti"],
            "u": ["ous"], "v": ["ive"], "z": ["ize"],
        }
        for suffix in table.get(ch, []):
            if self._ends(suffix):
                # -ion 只在 stem 以 s 或 t 結尾時才移除
                if suffix == "ion" and not (self.j >= 0 and self.b[self.j] in "st"):
                    continue
                if self._m() > 1:
                    self.k = self.j
                return

    def _step5(self):
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._double_cons(self.k) and self._m() > 1:
            self.k -= 1

    # ---- 對外介面 -------------------------------------------------
    def stem(self, word):
        word = word.lower()
        if len(word) <= 2:          # 長度 <= 2 的字不做處理
            return word
        self.b = word
        self.k = len(word) - 1
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


from functools import lru_cache

@lru_cache(maxsize=50000)
def stem(word):
    """Thread-safe Porter stemming; preserve non-ASCII/compound tokens."""
    word = word.casefold()
    if not word.isascii() or not word.isalpha():
        return word
    return PorterStemmer().stem(word)
