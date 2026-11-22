"""Recherche de la question la plus proche dans la base de connaissances (sans bibliothèque externe)."""
import math, re, unicodedata
from collections import Counter

STOP = set("""le la les un une des de du d l a au aux et ou en que qui quoi quand est il elle ils je j tu vous nous mon ma mes
ton ta tes son sa ses ce cet cette ces pas ne n se s y dans pour par sur avec sans puis peut peux dois doit faire fait ai as
avez ont suis etre avoir si plus tres bien c qu comment quel quelle quels quelles ca cela ceci est-ce""".split())


def norm(texte):
    t = unicodedata.normalize("NFD", texte.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[’']", " ", t)


def tokens(texte):
    return [w[:5] for w in re.findall(r"[a-z0-9]+", norm(texte)) if w not in STOP and len(w) > 1]


class Index:
    def __init__(self, fiches):
        self.fiches = fiches
        docs = [tokens(f["question"]) * 2 + tokens(f["reponse"]) for f in fiches]
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norme = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norme for t, x in v.items()}

    def chercher(self, question, n=3, seuil=0.25, scores=False):
        q = self._vec([t for t in tokens(question) if t in self.idf])
        if not q:
            return []
        notes = sorted(((sum(w * v.get(t, 0) for t, w in q.items()), f)
                        for v, f in zip(self.vecs, self.fiches)), key=lambda x: -x[0])
        res, vus = [], set()
        for s, f in notes:
            if s < seuil or len(res) >= n:
                break
            if f["reponse"] in vus:
                continue
            vus.add(f["reponse"])
            res.append((s, f) if scores else f)
        return res
