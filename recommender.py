#!/usr/bin/env python3
"""
Indian Recipe Recommender
=========================
Content-based recommender over the Archana's Kitchen "Indian Food" dataset.

Pipeline
--------
1. Parse the messy free-text ingredient lists into canonical ingredient tokens
   (quantities/units/prep words stripped, Hindi/English synonyms merged:
   jeera = cumin, haldi = turmeric, dahi = curd = yogurt ...).
2. Fit a TF-IDF model over those tokens, so rare, defining ingredients
   (paneer, kasuri methi) count for more than common ones (onion, cumin).
3. Score every recipe against what you have with a hybrid of
      cosine similarity (TF-IDF)  +  IDF-weighted pantry coverage
   and report exactly what is still missing.
4. Dietary rules (vegetarian, vegan, jain, gluten-free ...) are derived from the
   parsed ingredients, because the dataset's own `Diet` column is sparse/noisy.

Usage
-----
  python recommender.py --have "paneer, tomato, onion, cream" --diet vegetarian
  python recommender.py --have "potato, peas, rice" --pantry-basics --max-time 30
  python recommender.py --similar "Masala Karela Recipe"
  python recommender.py --evaluate
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_CSV = "IndianFoodDatasetCSV.csv"

# --------------------------------------------------------------------------- #
# 1. Ingredient parsing
# --------------------------------------------------------------------------- #
# Ordered: first match wins, so specific patterns come before general ones.
_RULES_RAW: list[tuple[str, str | tuple[str, ...]]] = [
    (r"ginger[\s-]*garlic", ("ginger", "garlic")),
    (r"\boil\b|\bghee\b.*\boil\b", "oil"),
    # meats / fish / egg
    (r"\bchicken\b", "chicken"),
    (r"\b(mutton|lamb|goat)\b", "mutton"),
    (r"\b(prawns?|shrimps?)\b", "prawn"),
    (r"\b(fish|salmon|tuna|rohu|pomfret|basa|surmai|hilsa|mackerel|sardines?|bangda|kingfish|seer)\b", "fish"),
    (r"\b(beef|pork|bacon|ham|sausages?|salami|turkey|duck|crab|lobster)\b", None),  # keep own word
    (r"\b(mince|minced meat|keema)\b", "minced meat"),
    (r"\beggs?\b", "egg"),
    # dairy
    (r"\bpaneer\b|cottage cheese", "paneer"),
    (r"\bcheese\b", "cheese"),
    (r"\b(curd|yogh?urt|dahi)\b", "yogurt"),
    (r"\b(buttermilk|chaas|chaach)\b", "buttermilk"),
    (r"coconut milk", "coconut milk"),
    (r"condensed milk", "condensed milk"),
    (r"\b(khoya|mawa|khoa)\b", "khoya"),
    (r"\bghee\b", "ghee"),
    (r"peanut butter", "peanut"),
    (r"\bbutter\b", "butter"),
    (r"\b(cream|malai)\b", "cream"),
    (r"\bmilk\b", "milk"),
    # alliums, aromatics
    (r"\b(spring onions?|green onions?|scallions?|onion greens)\b", "spring onion"),
    (r"onion seeds|nigella|kalonji", "kalonji"),
    (r"\b(onions?|pyaz|shallots?)\b", "onion"),
    (r"\bgarlic\b|lehsun", "garlic"),
    (r"\bginger\b|adrak", "ginger"),
    (r"curry leaves|kadi patta|kari patta", "curry leaves"),
    (r"\b(mint|pudina)\b", "mint"),
    (r"\b(coriander leaves|cilantro|dhania leaves|coriander)\b", None),  # resolved below
    # spices
    (r"coriander (powder|seeds?)|dhania (powder|seeds?)|dhania jeera", "coriander powder"),
    (r"\bturmeric|haldi\b", "turmeric"),
    (r"\b(cumin|jeera|shahi jeera)\b", "cumin"),
    (r"kashmiri|red chill?i powder|chill?i powder|lal mirch|paprika|cayenne", "red chilli powder"),
    (r"chill?i flakes", "chilli flakes"),
    (r"chill?i sauce|chilli garlic", "chilli sauce"),
    (r"dry red chill?i|dried red chill?i|whole red chill?i|sukhi lal|red chill?i", "dry red chilli"),
    (r"chill?i|hari mirch|serrano|jalapeno", "green chilli"),
    (r"garam masala", "garam masala"),
    (r"chaat masala", "chaat masala"),
    (r"sambh?ar (powder|masala)", "sambar powder"),
    (r"rasam (powder|masala)", "rasam powder"),
    (r"\b(amchur|amchoor|dry mango powder)\b", "amchur"),
    (r"\b(hing|asafoetida|asafetida)\b", "asafoetida"),
    (r"\b(mustard|sarson)\b|\brai\b", "mustard seeds"),
    (r"kasuri methi|dried fenugreek", "kasuri methi"),
    (r"fenugreek seeds?|methi (dana|seeds)", "fenugreek seeds"),
    (r"fenugreek|\bmethi\b", "fenugreek leaves"),
    (r"black pepper|peppercorns?|kali mirch|pepper powder|\bpepper\b", "black pepper"),
    (r"\b(cardamom|elaichi)\b", "cardamom"),
    (r"\b(cinnamon|dalchini)\b", "cinnamon"),
    (r"\b(cloves?|laung)\b", "cloves"),
    (r"bay leaf|bay leaves|tej patta", "bay leaf"),
    (r"star anise", "star anise"),
    (r"\b(fennel|saunf)\b", "fennel"),
    (r"\b(saffron|kesar)\b", "saffron"),
    (r"\b(nutmeg|jaiphal)\b", "nutmeg"),
    (r"\b(carom|ajwain)\b", "ajwain"),
    (r"poppy seeds|khus khus", "poppy seeds"),
    (r"\b(sesame|til)\b", "sesame"),
    (r"\bsalt\b|rock salt|sendha namak|black salt|kala namak", "salt"),
    (r"\bwater\b", "water"),
    # staples / grains / flours / dals
    (r"\b(basmati|rice)\b(?!.*\b(flour|vermicelli)\b)", None),  # resolved below
    (r"rice flour", "rice flour"),
    (r"\b(poha|flattened rice|beaten rice|aval)\b", "poha"),
    (r"puffed rice|murmura|mamra", "puffed rice"),
    (r"wheat flour|\batta\b|whole wheat", "wheat flour"),
    (r"\b(maida|all[\s-]?purpose flour|refined flour|plain flour)\b", "all purpose flour"),
    (r"\b(besan|gram flour|chickpea flour)\b", "gram flour"),
    (r"\b(semolina|sooji|suji|rava|rawa)\b", "semolina"),
    (r"corn ?flour|corn ?starch", "cornflour"),
    (r"\b(oats|rolled oats)\b", "oats"),
    (r"broken wheat|dalia|daliya|bulgur", "broken wheat"),
    (r"\b(vermicelli|seviyan|sevai)\b", "vermicelli"),
    (r"\b(sabudana|sago|tapioca pearls)\b", "sabudana"),
    (r"\b(pasta|penne|macaroni|spaghetti|fusilli|lasagn?e)\b", "pasta"),
    (r"\bnoodles?\b", "noodles"),
    (r"\b(bread|pav|buns?|rusk|baguette|tortillas?|roti|chapati|naan|paratha|breadcrumbs)\b", "bread"),
    (r"\b(toor|tuvar|arhar|pigeon pea)", "toor dal"),
    (r"chana dal|bengal gram|split chickpea", "chana dal"),
    (r"\b(moong|mung|green gram)\b", "moong dal"),
    (r"\b(urad|black gram)\b", "urad dal"),
    (r"\b(masoor|red lentils?)\b", "masoor dal"),
    (r"chickpeas?|\bchole\b|kabuli|garbanzo|\bchana\b", "chickpeas"),
    (r"\b(rajma|kidney beans?)\b", "kidney beans"),
    (r"\b(dal|lentils?)\b", "dal"),
    # vegetables
    (r"sweet potato", "sweet potato"),
    (r"\b(potato(es)?|aloo)\b", "potato"),
    (r"\b(tomato(es)?|tamatar)\b(?!.*ketchup)", "tomato"),
    (r"ketchup", "tomato ketchup"),
    (r"\b(cauliflower|gobi|gobhi)\b", "cauliflower"),
    (r"\bcabbage\b", "cabbage"),
    (r"\b(carrots?|gajar)\b", "carrot"),
    (r"\b(capsicum|bell peppers?|shimla mirch)\b", "capsicum"),
    (r"\b(brinjal|eggplant|aubergine|baingan)\b", "brinjal"),
    (r"\b(okra|bhindi|ladies finger)\b", "okra"),
    (r"\b(spinach|palak)\b", "spinach"),
    (r"bottle gourd|\b(lauki|doodhi)\b", "bottle gourd"),
    (r"bitter gourd|karela|pavakkai", "bitter gourd"),
    (r"ridge gourd|turai|\btori\b", "ridge gourd"),
    (r"\bpumpkin\b", "pumpkin"),
    (r"\b(beetroot|beet)\b", "beetroot"),
    (r"\b(radish|mooli)\b", "radish"),
    (r"\bcucumber\b", "cucumber"),
    (r"\bmushrooms?\b", "mushroom"),
    (r"\b(sweet corn|corn|bhutta)\b", "corn"),
    (r"\b(peas|matar)\b", "peas"),
    (r"\b(french beans|green beans|beans)\b", "beans"),
    (r"raw banana|plantain", "raw banana"),
    (r"\b(yam|suran|senai|ratalu)\b", "yam"),
    (r"\bdrumsticks?\b", "drumstick"),
    (r"\b(zucchini|courgette)\b", "zucchini"),
    (r"\bbroccoli\b", "broccoli"),
    (r"\bbasil\b", "basil"),
    # souring / sweetening / misc
    (r"\b(lemon|lime|nimbu)\b", "lemon"),
    (r"\b(tamarind|imli)\b", "tamarind"),
    (r"\bcoconut\b|\bnariyal\b", "coconut"),
    (r"\b(jaggery|gur)\b", "jaggery"),
    (r"\bsugar\b", "sugar"),
    (r"\bhoney\b", "honey"),
    (r"\bvinegar\b", "vinegar"),
    (r"soy sauce|soya sauce", "soy sauce"),
    (r"baking soda|cooking soda|bicarbonate", "baking soda"),
    (r"baking powder", "baking powder"),
    (r"\byeast\b", "yeast"),
    # nuts
    (r"\b(cashews?|kaju)\b", "cashew"),
    (r"\b(almonds?|badam)\b", "almond"),
    (r"\b(pistachios?|pista)\b", "pistachio"),
    (r"\b(walnuts?|akhrot)\b", "walnut"),
    (r"\b(peanuts?|groundnuts?|moongphali)\b", "peanut"),
    (r"\b(raisins?|kishmish)\b", "raisin"),
]


def _compile_rules():
    out = []
    for pat, canon in _RULES_RAW:
        out.append((re.compile(pat, re.I), canon))
    return out


RULES = _compile_rules()
CANONICAL = {c for _, c in _RULES_RAW if isinstance(c, str)} | {
    "ginger", "garlic", "coriander leaves", "rice", "beef", "pork", "bacon", "ham",
    "sausage", "salami", "turkey", "duck", "crab", "lobster",
}

# Things almost every kitchen has; never counted as "missing".
STAPLES = {"salt", "water", "oil"}
# --pantry-basics: a typical Indian masala-dabba + staples.
PANTRY_BASICS = STAPLES | {
    "turmeric", "red chilli powder", "cumin", "coriander powder", "garam masala",
    "mustard seeds", "asafoetida", "sugar", "black pepper", "ghee",
}

_UNITS = (
    r"(?:cups?|tablespoons?|teaspoons?|tbsp|tsp|grams?|gms?|g|kgs?|kilograms?|ml|litres?|liters?|"
    r"inch(?:es)?|sprigs?|pinch(?:es)?|bunch(?:es)?|handfuls?|dash(?:es)?|stalks?|pieces?|slices?|"
    r"cans?|packets?|nos?)"
)
_DESC = re.compile(
    r"\b(?:fresh|freshly|finely|roughly|thinly|coarsely|chopped|sliced|grated|crushed|ground|diced|"
    r"minced|boiled|cooked|roasted|ripe|raw|large|medium|small|big|tiny|hot|warm|cold|soaked|"
    r"peeled|washed|cleaned|optional|powdered|whole|dried|dry)\b"
)


def split_items(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    items, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    return [i.strip() for i in items if i.strip()]


def _normalize(raw: str) -> tuple[str, str]:
    t = raw.lower().strip()
    t = re.split(r"\s+[-–]\s+", t)[0]                       # drop " - to taste / - chopped"
    aliases = " ".join(re.findall(r"\(([^)]*)\)", t))      # (Haldi) -> alias text
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9/.\- ]", " ", t)
    t = re.sub(r"^[\d\s/.\-]+", "", t)                      # leading quantity
    for _ in range(2):
        t = re.sub(rf"^{_UNITS}\b\s*", "", t)
        t = re.sub(r"^of\s+", "", t)
    t = re.sub(r"^cloves\s+(?=\w)", "", t)                 # "4 cloves garlic" -> garlic
    t = re.sub(r"\s+", " ", t).strip()
    return t, aliases.lower()


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def canonicalize(raw: str) -> list[str]:
    """Free-text ingredient line -> list of canonical tokens (usually one)."""
    main, alias = _normalize(raw)
    if not main and not alias:
        return []
    for text in (main, f"{main} {alias}".strip()):
        for pat, canon in RULES:
            m = pat.search(text)
            if not m:
                continue
            if canon is None:                                # special own-word cases
                word = m.group(0).strip()
                if word in {"coriander leaves", "cilantro", "coriander", "dhania leaves"}:
                    return ["coriander leaves"]
                if word in {"basmati", "rice"} or "rice" in word:
                    return ["rice"]
                return [_singular(word.replace("sausages", "sausage"))]
            return list(canon) if isinstance(canon, tuple) else [canon]
    # Fallback: keep the cleaned phrase (minus prep adjectives) if it's short.
    fb = _DESC.sub(" ", main)
    fb = re.sub(r"\s+", " ", fb).strip()
    return [_singular(fb)] if 2 < len(fb) <= 30 else []


def parse_ingredients(text: str) -> list[str]:
    toks: list[str] = []
    for item in split_items(text):
        toks.extend(canonicalize(item))
    return list(dict.fromkeys(toks))                         # de-dupe, keep order


# --------------------------------------------------------------------------- #
# 2. Dietary rules (derived from ingredients, not from the noisy `Diet` column)
# --------------------------------------------------------------------------- #
MEAT = {"chicken", "mutton", "prawn", "fish", "minced meat", "beef", "pork", "bacon",
        "ham", "sausage", "salami", "turkey", "duck", "crab", "lobster"}
DAIRY = {"milk", "yogurt", "paneer", "ghee", "butter", "cream", "cheese", "buttermilk",
         "condensed milk", "khoya"}
GLUTEN = {"wheat flour", "all purpose flour", "semolina", "bread", "pasta", "noodles",
          "vermicelli", "broken wheat", "oats", "soy sauce"}
NUTS = {"cashew", "almond", "pistachio", "walnut", "peanut"}
ALLIUM = {"onion", "garlic", "spring onion"}
JAIN_NO = ALLIUM | {"potato", "carrot", "radish", "beetroot", "sweet potato", "yam",
                    "ginger", "mushroom"}

DIETS: dict[str, set[str]] = {
    "non-vegetarian": set(),
    "eggetarian": set(MEAT),
    "vegetarian": MEAT | {"egg"},
    "vegan": MEAT | {"egg", "honey"} | DAIRY,
    "jain": MEAT | {"egg"} | JAIN_NO,
    "sattvic": MEAT | {"egg"} | ALLIUM,       # "No Onion No Garlic"
    "gluten-free": set(GLUTEN),
    "dairy-free": set(DAIRY),
    "nut-free": set(NUTS),
}
_NAME_MEAT = re.compile(r"\b(chicken|mutton|lamb|fish|prawns?|shrimp|keema|beef|pork|egg|eggs)\b", re.I)


# --------------------------------------------------------------------------- #
# 3. The model
# --------------------------------------------------------------------------- #
def _identity(x):  # module-level so the vectorizer stays picklable
    return x


class RecipeRecommender:
    def __init__(self, csv_path: str | Path = DEFAULT_CSV, alpha: float = 0.5, support: float = 4.0):
        """alpha = weight of TF-IDF cosine; (1-alpha) = weight of pantry coverage.
        support = damping so recipes matching only one trivial ingredient do not win."""
        self.alpha = alpha
        self.support = support
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["TranslatedIngredients"]).copy()
        for c in ("TranslatedRecipeName", "Course", "Cuisine", "URL"):
            df[c] = df[c].astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        # ~8% of rows have Hindi-script ingredient lists (never translated): drop them.
        nonascii = df["TranslatedIngredients"].map(lambda x: sum(ord(c) > 127 for c in x) / max(1, len(x)))
        self.n_dropped_hindi = int((nonascii > 0.1).sum())
        df = df[nonascii <= 0.1]
        df = df.drop_duplicates(subset="TranslatedRecipeName").reset_index(drop=True)

        raw_tokens = [parse_ingredients(t) for t in df["TranslatedIngredients"]]

        # Drop one-off fallback tokens (typos, garbled lines) so they don't skew IDF.
        counts: dict[str, int] = {}
        for toks in raw_tokens:
            for t in set(toks):
                counts[t] = counts.get(t, 0) + 1
        keep = lambda t: t in CANONICAL or counts[t] >= 3
        self.tags_tokens = [[t for t in toks if keep(t)] for toks in raw_tokens]
        # Scoring tokens exclude universal staples.
        self.tokens = [[t for t in toks if t not in STAPLES] for toks in self.tags_tokens]

        good = np.array([len(t) >= 2 for t in self.tokens])
        self.n_dropped_sparse = int((~good).sum())
        df = df[good].reset_index(drop=True)
        self.tags_tokens = [t for t, g in zip(self.tags_tokens, good) if g]
        self.tokens = [t for t, g in zip(self.tokens, good) if g]

        self.df = df
        self.vec = TfidfVectorizer(analyzer=_identity, norm="l2")
        self.X = self.vec.fit_transform(self.tokens).tocsr()
        self.vocab = self.vec.vocabulary_
        self.idf = self.vec.idf_
        self.B = self.X.copy()
        self.B.data[:] = 1.0
        self.den = np.asarray(self.B @ self.idf).ravel()      # total IDF mass per recipe

        # Pre-compute which dietary rules each recipe satisfies.
        self._diet_ok: dict[str, np.ndarray] = {}
        name_map = {"lamb": "mutton", "shrimp": "prawn", "prawns": "prawn", "keema": "minced meat",
                    "eggs": "egg"}
        name_meat = [{name_map.get(m.lower(), m.lower()) for m in _NAME_MEAT.findall(n)}
                     for n in df["TranslatedRecipeName"]]
        full = [set(t) | nm for t, nm in zip(self.tags_tokens, name_meat)]
        for diet, forbidden in DIETS.items():
            self._diet_ok[diet] = np.array([not (forbidden & s) for s in full])
        self._full_tokens = full

    # ----- query handling ------------------------------------------------- #
    def resolve_query(self, have: str | list[str]) -> tuple[list[str], list[str], dict[str, str]]:
        """Return (known tokens, unknown tokens, auto-corrections)."""
        items = split_items(have) if isinstance(have, str) else list(have)
        tokens: list[str] = []
        for it in items:
            tokens.extend(canonicalize(it))
        known, unknown, fixed = [], [], {}
        for t in dict.fromkeys(tokens):
            if t in self.vocab:
                known.append(t)
                continue
            close = difflib.get_close_matches(t, self.vocab.keys(), n=1, cutoff=0.82)
            if close:
                fixed[t] = close[0]
                known.append(close[0])
            else:
                unknown.append(t)
        return list(dict.fromkeys(known)), unknown, fixed

    def _mask(self, tokens, universe=None) -> np.ndarray:
        m = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                m[self.vocab[t]] = 1.0
        return m

    # ----- recommendation ------------------------------------------------- #
    def recommend(
        self,
        have: str | list[str],
        diet: list[str] | None = None,
        exclude: str | list[str] | None = None,
        max_time: int | None = None,
        course: str | None = None,
        cuisine: str | None = None,
        max_missing: int | None = None,
        pantry_basics: bool = False,
        min_match: int = 1,
        top_k: int = 10,
        alpha: float | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        alpha = self.alpha if alpha is None else alpha
        known, unknown, fixed = self.resolve_query(have)
        info = {"understood": known, "unknown": unknown, "corrected": fixed}
        if not known:
            return pd.DataFrame(), info

        explicit = self._mask(known)
        assumed = PANTRY_BASICS if pantry_basics else STAPLES
        have_mask = np.maximum(explicit, self._mask(assumed))

        q = self.vec.transform([[t for t in known if t not in STAPLES]])
        cos = (self.X @ q.T).toarray().ravel()
        num = np.asarray(self.B @ (self.idf * have_mask)).ravel()
        cov = np.divide(num, self.den, out=np.zeros_like(num), where=self.den > 0)
        overlap = np.asarray(self.B @ explicit).ravel()
        matched = np.asarray(self.B @ (self.idf * explicit)).ravel()   # IDF mass of *your* ingredients used
        score = (alpha * cos + (1 - alpha) * cov) * (matched / (matched + self.support))

        ok = overlap >= min_match
        for d in diet or []:
            ok &= self._diet_ok[d]
        if exclude:
            ex = [t for it in (split_items(exclude) if isinstance(exclude, str) else exclude)
                  for t in canonicalize(it)]
            if ex:
                ok &= np.array([not (set(ex) & s) for s in self._full_tokens])
        if max_time:
            t = self.df["TotalTimeInMins"].to_numpy()
            ok &= (t > 0) & (t <= max_time)
        if course:
            ok &= self.df["Course"].str.contains(course, case=False, regex=False).to_numpy()
        if cuisine:
            ok &= self.df["Cuisine"].str.contains(cuisine, case=False, regex=False).to_numpy()

        n_missing = np.asarray(self.B.sum(axis=1)).ravel() - np.asarray(self.B @ have_mask).ravel()
        if max_missing is not None:
            ok &= n_missing <= max_missing

        idx = np.where(ok)[0]
        idx = idx[np.argsort(-score[idx], kind="stable")][:top_k]

        rows = []
        for i in idx:
            toks = self.tokens[i]
            have_t = [t for t in toks if have_mask[self.vocab[t]] and explicit[self.vocab[t]]]
            miss_t = [t for t in toks if not have_mask[self.vocab[t]]]
            miss_t.sort(key=lambda t: -self.idf[self.vocab[t]])   # defining ingredients first
            r = self.df.iloc[i]
            rows.append({
                "recipe": r["TranslatedRecipeName"], "score": round(float(score[i]), 3),
                "coverage": round(float(cov[i]), 2), "you_have": have_t, "missing": miss_t,
                "minutes": int(r["TotalTimeInMins"]), "course": r["Course"],
                "cuisine": r["Cuisine"], "servings": int(r["Servings"]), "url": r["URL"],
            })
        return pd.DataFrame(rows), info

    # ----- "more like this" ------------------------------------------------ #
    def similar(self, name: str, top_k: int = 8) -> pd.DataFrame:
        names = self.df["TranslatedRecipeName"]
        hit = names[names.str.lower() == name.lower()]
        if hit.empty:
            hit = names[names.str.contains(name, case=False, regex=False)]
        if hit.empty:
            close = difflib.get_close_matches(name, names.tolist(), n=3, cutoff=0.5)
            raise KeyError(f"No recipe named {name!r}. Closest: {close}")
        i = hit.index[0]
        sims = (self.X @ self.X[i].T).toarray().ravel()
        sims[i] = -1
        best = np.argsort(-sims)[:top_k]
        out = self.df.iloc[best][["TranslatedRecipeName", "TotalTimeInMins", "Course", "Cuisine", "URL"]].copy()
        out.insert(1, "similarity", np.round(sims[best], 3))
        return out.reset_index(drop=True), names[i]

    # ----- evaluation ------------------------------------------------------ #
    def evaluate_retrieval(self, n: int = 600, keep_frac: float = 0.6, seed: int = 0,
                           alphas=(0.0, 0.25, 0.35, 0.5, 0.75, 1.0), support: float | None = None) -> pd.DataFrame:
        support = self.support if support is None else support
        """
        Simulated user test: take a recipe, pretend you only own a random
        `keep_frac` of its (non-staple) ingredients, and see where the recipe
        lands in the ranking. Reports Hit@1/5/10 and MRR per alpha.
        (alpha=1 is pure TF-IDF cosine, alpha=0 is pure pantry coverage.)
        """
        rng = np.random.default_rng(seed)
        eligible = [i for i, t in enumerate(self.tokens) if len(t) >= 5]
        picks = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
        queries = []
        for i in picks:
            toks = self.tokens[i]
            k = max(2, int(round(len(toks) * keep_frac)))
            queries.append((i, list(rng.choice(toks, size=k, replace=False))))

        res = []
        for a in alphas:
            ranks = []
            for i, q_tokens in queries:
                explicit = self._mask(q_tokens)
                q = self.vec.transform([q_tokens])
                cos = (self.X @ q.T).toarray().ravel()
                num = np.asarray(self.B @ (self.idf * explicit)).ravel()
                cov = np.divide(num, self.den, out=np.zeros_like(num), where=self.den > 0)
                matched = np.asarray(self.B @ (self.idf * explicit)).ravel()
                s = (a * cos + (1 - a) * cov) * (matched / (matched + support))
                ranks.append(int((s > s[i]).sum()) + 1)
            r = np.array(ranks)
            res.append({"alpha": a, "Hit@1": (r <= 1).mean(), "Hit@5": (r <= 5).mean(),
                        "Hit@10": (r <= 10).mean(), "MRR": (1 / r).mean()})
        return pd.DataFrame(res).round(3)

    def evaluate_diet_rules(self) -> dict:
        """Compare ingredient-derived veg/non-veg/egg tags with the dataset's Diet label."""
        label = self.df["Diet"].str.lower()
        label_nv = label.str.contains("non veg").to_numpy()
        label_egg = label.str.contains("eggetarian").to_numpy()
        ours_nv = ~self._diet_ok["eggetarian"]
        ours_egg = self._diet_ok["eggetarian"] & ~self._diet_ok["vegetarian"]
        dis = np.where(label_nv != ours_nv)[0]
        return {
            "recipes": len(self.df),
            "non_veg_agreement": float((label_nv == ours_nv).mean()),
            "label_non_veg": int(label_nv.sum()), "rules_non_veg": int(ours_nv.sum()),
            "label_eggetarian": int(label_egg.sum()), "rules_eggetarian": int(ours_egg.sum()),
            "disagreements": [(self.df.at[i, "TranslatedRecipeName"], self.df.at[i, "Diet"],
                               "non-veg" if ours_nv[i] else "veg") for i in dis[:8]],
        }


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def _print_results(df: pd.DataFrame, info: dict):
    if info["corrected"]:
        print("Interpreted: " + ", ".join(f"'{a}' -> '{b}'" for a, b in info["corrected"].items()))
    if info["unknown"]:
        print("Not found in dataset (ignored): " + ", ".join(info["unknown"]))
    if df.empty:
        print("\nNo recipes match. Try fewer filters, add ingredients, or raise --max-missing.")
        return
    print(f"\nUsing: {', '.join(info['understood'])}\n")
    for n, r in enumerate(df.itertuples(), 1):
        t = f"{r.minutes} min" if r.minutes else "time n/a"
        print(f"{n:>2}. {r.recipe}   [{r.score:.2f}]  {t} · {r.course} · {r.cuisine}")
        print(f"     have   : {', '.join(r.you_have) or '-'}")
        print(f"     missing: {', '.join(r.missing) if r.missing else 'nothing - you can cook this now!'}")
        print(f"     {r.url}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Indian recipe recommender (TF-IDF + pantry coverage).")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--have", help='Ingredients you have, comma separated: "paneer, tomato, jeera"')
    p.add_argument("--diet", nargs="+", choices=sorted(DIETS), default=[],
                   help="One or more dietary rules, e.g. --diet vegetarian gluten-free")
    p.add_argument("--exclude", help='Ingredients/allergens to avoid: "peanut, cashew"')
    p.add_argument("--max-time", type=int, help="Max total minutes")
    p.add_argument("--course", help='e.g. Dinner, Snack, Dessert, "South Indian Breakfast"')
    p.add_argument("--cuisine", help='e.g. Punjabi, "South Indian", Bengali')
    p.add_argument("--max-missing", type=int, help="Only recipes missing at most N ingredients")
    p.add_argument("--pantry-basics", action="store_true",
                   help="Assume you own common masalas (haldi, jeera, garam masala, sugar, ghee...)")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--alpha", type=float, help="Weight of TF-IDF cosine vs coverage (0-1)")
    p.add_argument("--similar", metavar="RECIPE", help="Find recipes similar to this one")
    p.add_argument("--evaluate", action="store_true", help="Run offline evaluation")
    a = p.parse_args(argv)

    if not Path(a.csv).exists():
        sys.exit(f"Dataset not found: {a.csv}  (use --csv path/to/IndianFoodDatasetCSV.csv)")
    rec = RecipeRecommender(a.csv)

    if a.evaluate:
        print("Retrieval test (simulated pantry = random 60% of a recipe's ingredients):")
        print(rec.evaluate_retrieval().to_string(index=False))
        print("\nDiet-rule check vs. dataset `Diet` label:")
        for k, v in rec.evaluate_diet_rules().items():
            print(f"  {k}: {v}")
        return
    if a.similar:
        out, matched = rec.similar(a.similar, a.top)
        print(f"Recipes similar to: {matched}\n")
        print(out.to_string(index=False))
        return
    if not a.have:
        p.error("provide --have, --similar or --evaluate")
    df, info = rec.recommend(a.have, a.diet, a.exclude, a.max_time, a.course, a.cuisine,
                             a.max_missing, a.pantry_basics, top_k=a.top, alpha=a.alpha)
    _print_results(df, info)


if __name__ == "__main__":
    main()
