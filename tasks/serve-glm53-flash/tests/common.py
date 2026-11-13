"""Shared logic of the accuracy gate: data loading, answer extraction and per-benchmark
equality. Used by score.py and serving_eval.py so grading and recomputation agree exactly."""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = "ABCDEFGHIJ"


def load_json(name):
    p = name if os.path.isabs(name) else os.path.join(HERE, name)
    with open(p) as f:
        return json.load(f)


# ---------------- extraction helpers ----------------
def _last_number(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return nums[-1] if nums else None


def gsm8k_gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def gsm8k_pred(text):
    if text is None:
        return None
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.findall(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if m:
        return m[-1].replace(",", "")
    return _last_number(text)


def extract_boxed(text):
    if text is None:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth, j, buf = 0, i, []
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
            if depth > 1:
                buf.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(c)
        else:
            buf.append(c)
        j += 1
    return "".join(buf).strip()


def norm_math(s):
    if s is None:
        return None
    s = s.strip()
    # keep \text{..}/\mathrm{..} CONTENT (deleting it would collapse text answers to "")
    s = re.sub(r"\\(?:text|mathrm|mbox|textbf|textit|textrm|operatorname)\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    for t in ("\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad", "\\\\", "$", "\\$"):
        s = s.replace(t, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" ", "").rstrip(".")
    return s.lower()


# ---- canonical MATH answer equivalence (Hendrycks MATH is_equiv) ----
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a); b = int(b)
        if string != "{}/{}".format(a, b):
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string

def _strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    # keep \text{..} content, lowercased, so text answers stay distinct
    string = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^}]*)\}", r"\1", string)
    return string.strip().lower()

def is_equiv(str1, str2):
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str(str1)) == _strip_string(str(str2))
    except Exception:
        return str(str1).strip() == str(str2).strip()


def mmlu_pred(text, n_opts=10):
    if text is None:
        return None
    pats = [r"answer is\s*\(?([A-J])\)?", r"answer:\s*\(?([A-J])\)?",
            r"\\boxed\{([A-J])\}", r"\b([A-J])\b\s*$", r"\(([A-J])\)"]
    for p in pats:
        m = re.findall(p, text)
        if m:
            for c in reversed(m):
                if LETTERS.index(c) < n_opts:
                    return c
    return None


def canary_pred(text):
    """Canaries are synthetic GSM8K-style word problems that elicit step-by-step
    reasoning ending in '#### <number>'. Reuse the GSM8K extractor: the chain of
    thought stabilizes the final numeric answer run-to-run (vs a bare guess)."""
    return gsm8k_pred(text)


# ---------------- unified extraction by benchmark ----------------
def extract(benchmark, text):
    if benchmark == "gsm8k":
        return gsm8k_pred(text)
    if benchmark == "mmlu_pro":
        return mmlu_pred(text)
    if benchmark == "math_500":
        return extract_boxed(text)
    if benchmark == "canary":
        return canary_pred(text)
    raise ValueError(benchmark)


# ---------------- per-benchmark equality ----------------
def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def answers_equal(benchmark, a, b):
    """True if answers a and b (a prediction and its gold answer) are the same answer
    for this benchmark."""
    if a is None or b is None:
        return False
    if benchmark in ("gsm8k",):
        return _num_eq(a, b)
    if benchmark == "canary":
        # numeric if both parse as numbers, else case-insensitive token match
        if _num_eq(a, b):
            return True
        return str(a).strip().lower() == str(b).strip().lower()
    if benchmark == "math_500":
        return is_equiv(a, b) or _num_eq(a, b)
    # mmlu_pro letters
    return str(a).strip() == str(b).strip()
