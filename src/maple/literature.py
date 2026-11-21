"""Reference verification against Crossref, OpenAlex and arXiv, and BibTeX generation.

Bibliographic metadata (authors, venue, DOI) is taken only from the services; nothing is typed by hand.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

UA = {"User-Agent": "maple-literature-verifier/0.1 (research reproducibility script)"}
_CONF_HINTS = ("ICLR", "ICML", "NeurIPS", "COLT", "KDD", "ECCV", "IJCAI", "CoRL", "IROS", "IDEAL", "MCS", "MLSys",
               "AutoML", "D&B")


def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"<[^>]+>", " ", t)  # strip HTML/MathML tags
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def is_match(title: str, year: int, cand_title: str, cand_year: int | None, thr: float = 0.92) -> bool:
    if cand_year is None or abs(int(cand_year) - int(year)) > 1:
        return False
    return title_similarity(title, cand_title) >= thr


class CachedClient:
    def __init__(self, cache_dir: Path, delay: float = 0.15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

    def get(self, url: str) -> str | None:
        import hashlib

        f = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
        if f.exists():
            return f.read_text(encoding="utf-8")
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=30)
                if r.status_code == 200:
                    f.write_text(r.text, encoding="utf-8")
                    time.sleep(self.delay)
                    return r.text
                if r.status_code in (429, 503):
                    time.sleep(2 + 3 * attempt)
                    continue
                return None
            except requests.RequestException:
                time.sleep(2 + 3 * attempt)
        return None


def crossref_title_candidates(item: dict) -> list[str]:
    t = (item.get("title") or [""])[0]
    out = [t]
    for sub in item.get("subtitle") or []:
        out += [f"{t}: {sub}", f"{t} {sub}"]
    return out


def verify_source_text(text: str, title: str, authors: list[str]) -> bool:
    """True if the title (normalised) and every author name occur in a trusted source's text."""
    nt = normalize_title(text)
    return normalize_title(title) in nt and all(normalize_title(a) in nt for a in authors)


_GENERIC_VENUE = ("arxiv", "openreview", "hal", "pubmed", "repository", "ebooks", "zenodo", "research square",
                  "cornell university", "neural information processing", "jmlr", "europe pmc")


def is_generic_venue(v: str) -> bool:
    """Repository/aggregator labels that do not name the publication venue."""
    return (not v) or any(g in v.lower() for g in _GENERIC_VENUE) or v.strip().lower() in ("hal",)


def clean_author(name: str) -> str:
    """Repair service artefacts such as 'PedregosaFabian' (family+given concatenated without a space)."""
    n = " ".join(name.split())
    if " " not in n and "," not in n:
        m = re.match(r"^([A-Z][a-zÀ-ſ'\-]+)([A-Z].*)$", n)
        if m:
            return f"{m.group(1)}, {m.group(2)}"
    return n


def doi_title_ok(title: str, service_title: str) -> bool:
    nt, ns = normalize_title(title), normalize_title(service_title)
    return title_similarity(title, service_title) >= 0.92 or (len(nt) > 20 and nt in ns)


def _crossref_item_year(it: dict) -> int | None:
    for f in ("published-print", "published-online", "issued", "published"):
        if it.get(f, {}).get("date-parts") and it[f]["date-parts"][0][0]:
            return it[f]["date-parts"][0][0]
    return None


def _crossref_record(it: dict) -> dict:
    authors = [f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip()
               for a in it.get("author", []) if a.get("family")]
    if not authors:
        authors = [a.get("name", "") for a in it.get("author", []) if a.get("name")]
    title = max(crossref_title_candidates(it)[:2], key=len)
    return {"title": title, "year": _crossref_item_year(it), "authors": authors, "doi": it.get("DOI"),
            "venue": (it.get("container-title") or [""])[0], "volume": it.get("volume"), "pages": it.get("page"),
            "publisher": it.get("publisher"), "crossref_type": it.get("type"),
            "url": f"https://doi.org/{it.get('DOI')}", "via": "crossref"}


def _crossref_doi(client: CachedClient, doi: str, title: str, year: int) -> dict | None:
    txt = client.get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not txt:
        return None
    it = json.loads(txt)["message"]
    if any(doi_title_ok(title, t) for t in crossref_title_candidates(it)):
        rec = _crossref_record(it)
        rec["via"] = "crossref-doi"
        return rec
    return None


def _crossref(client: CachedClient, title: str, year: int) -> dict | None:
    url = "https://api.crossref.org/works?rows=8&query.bibliographic=" + urllib.parse.quote(title)
    txt = client.get(url)
    if not txt:
        return None
    for it in json.loads(txt)["message"]["items"]:
        if not any(is_match(title, year, t, _crossref_item_year(it)) for t in crossref_title_candidates(it)):
            continue
        if it.get("type") in ("peer-review", "component", "posted-content", "dataset") and "arxiv" not in it.get("DOI", ""):
            continue
        return _crossref_record(it)
    return None


def _openalex(client: CachedClient, title: str, year: int) -> dict | None:
    url = ("https://api.openalex.org/works?per-page=8&search=" + urllib.parse.quote(title))
    txt = client.get(url)
    if not txt:
        return None
    for it in json.loads(txt).get("results", []):
        if not is_match(title, year, it.get("title") or "", it.get("publication_year")):
            continue
        authors = [a["author"]["display_name"] for a in it.get("authorships", [])]
        src = ((it.get("primary_location") or {}).get("source") or {})
        doi = (it.get("doi") or "").replace("https://doi.org/", "") or None
        return {"title": it["title"], "year": it["publication_year"], "authors": authors, "doi": doi,
                "venue": src.get("display_name") or "", "openalex": it.get("id"), "openalex_type": it.get("type"),
                "url": it.get("doi") or (it.get("primary_location") or {}).get("landing_page_url"), "via": "openalex"}
    return None


def _arxiv(client: CachedClient, title: str, year: int, arxiv_id: str | None) -> dict | None:
    if arxiv_id:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    else:
        q = urllib.parse.quote(f'ti:"{normalize_title(title)}"')
        url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    txt = client.get(url)
    if not txt:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(txt)
    for e in root.findall("a:entry", ns):
        t = " ".join((e.findtext("a:title", "", ns)).split())
        pub = e.findtext("a:published", "", ns)
        y = int(pub[:4]) if pub else None
        # arXiv preprint year may precede venue year by >1; accept when the id was given explicitly
        if not (is_match(title, year, t, y) or (arxiv_id and title_similarity(title, t) >= 0.92)):
            continue
        aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        return {"title": t, "year": y, "authors": authors, "arxiv": aid, "url": f"https://arxiv.org/abs/{aid}",
                "via": "arxiv"}
    return None


def verify(client: CachedClient, cand: dict) -> dict:
    """Return a verified record (or one with verified=False)."""
    title, year = cand["title"], int(cand["year"])
    rec: dict = {"key": cand["key"], "query_title": title, "query_year": year, "verified": False}
    if cand.get("source_url"):
        txt = client.get(cand["source_url"])
        if txt and verify_source_text(txt, title, cand.get("authors", [])):
            rec.update({"title": title, "year": year, "authors": cand["authors"], "venue": cand.get("venue_hint", ""),
                        "url": cand.get("url") or cand["source_url"], "via": "source_text", "verified": True,
                        "verified_via": f"source_text:{cand['source_url']}", "venue_source": "source_text",
                        "entry_type": _entry_type({"via": "source_text"}, cand)})
        return rec
    cr = _crossref_doi(client, cand["doi"], title, year) if cand.get("doi") else None
    cr = cr or _crossref(client, title, year)
    oa = None if cr else _openalex(client, title, year)
    ax = _arxiv(client, title, year, cand.get("arxiv")) if cand.get("arxiv") else None
    best = cr or oa or ax
    if not best:
        return rec
    rec.update(best)
    rec["verified"] = True
    rec["verified_via"] = best["via"] + ("+arxiv" if ax and best is not ax else "")
    if ax:
        rec["arxiv"] = ax["arxiv"]
    # venue: service value unless missing or only an arXiv/repository label
    v = rec.get("venue") or ""
    if is_generic_venue(v):
        hint = cand.get("venue_hint", "")
        if hint and hint != "arXiv":
            rec["venue"] = _expand_venue(hint)
            rec["venue_source"] = "annotator_hint"
        elif not v:
            rec["venue"] = "arXiv preprint"
            rec["venue_source"] = "arxiv"
        else:
            rec["venue_source"] = "service"
    else:
        rec["venue_source"] = "service"
    rec["authors"] = [clean_author(a) for a in rec.get("authors", [])]
    rec["entry_type"] = _entry_type(rec, cand)
    # Conference papers indexed only via their arXiv preprint carry the preprint year; use the venue year.
    tol = 4 if cand.get("arxiv") else 1  # an explicit arXiv id pins the paper; preprints can precede venues by years
    if rec["venue_source"] == "annotator_hint" and rec.get("year") != year and abs(int(rec["year"]) - year) <= tol:
        rec["preprint_year"] = rec["year"]
        rec["year"] = year
        rec["year_source"] = "venue_year(annotator)"
    return rec


_VENUES = {
    "ICLR": "International Conference on Learning Representations (ICLR)",
    "ICML": "Proceedings of the International Conference on Machine Learning (ICML)",
    "NeurIPS": "Advances in Neural Information Processing Systems (NeurIPS)",
    "NeurIPS D&B": "Advances in Neural Information Processing Systems, Datasets and Benchmarks Track",
    "COLT": "Proceedings of the Annual Workshop on Computational Learning Theory (COLT)",
    "CoRL": "Proceedings of the Conference on Robot Learning (CoRL)",
    "MLSys": "Proceedings of Machine Learning and Systems (MLSys)",
    "JMLR": "Journal of Machine Learning Research",
    "IDEAL": "Intelligent Data Engineering and Automated Learning (IDEAL), Lecture Notes in Computer Science",
    "KDD": "Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
    "ECCV": "European Conference on Computer Vision (ECCV), Lecture Notes in Computer Science",
    "IJCAI": "Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)",
}


def _expand_venue(h: str) -> str:
    return _VENUES.get(h, h)


def _entry_type(rec: dict, cand: dict) -> str:
    ct = rec.get("crossref_type") or ""
    if ct == "journal-article":
        return "article"
    if ct in ("proceedings-article", "book-chapter"):
        return "inproceedings"
    if ct in ("book", "monograph"):
        return "book"
    hint = cand.get("venue_hint", "")
    if any(h in hint for h in _CONF_HINTS):
        return "inproceedings"
    if hint == "arXiv" or rec.get("via") == "arxiv":
        return "misc"
    if (rec.get("openalex_type") or "") == "article" and rec.get("venue_source") == "service":
        return "article"
    return "article" if hint else "misc"


def _bib_name(n: str) -> str:
    n = " ".join(n.split())
    if "," in n:
        return n
    parts = n.split(" ")
    if len(parts) == 1:
        return n
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _esc(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&amp;", "&").replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_") \
        .replace("#", "\\#")


_TEX_SPECIAL = {"\u00df": "{\\ss}", "\u00e6": "{\\ae}", "\u00c6": "{\\AE}", "\u00f8": "{\\o}", "\u00d8": "{\\O}",
                "\u0141": "{\\L}", "\u0142": "{\\l}", "\u0153": "{\\oe}", "\u0131": "{\\i}", "\u00a0": "~",
                "\u2013": "--", "\u2014": "---", "\u2010": "-", "\u2011": "-", "\u2018": "`", "\u2019": "'",
                "\u201c": "``", "\u201d": "''"}
_TEX_ACCENT = {"\u0301": "'", "\u0300": "`", "\u0308": '"', "\u0302": "^", "\u0303": "~", "\u0304": "=",
               "\u0307": ".", "\u0327": "c ", "\u030c": "v ", "\u030a": "r ", "\u0306": "u ", "\u030b": "H ",
               "\u0328": "k "}


def tex_ascii(s: str) -> str:
    """Non-ASCII text -> ASCII LaTeX (accents as {\\"u}); BibTeX is 8-bit and can split UTF-8 when wrapping lines."""
    import unicodedata
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in _TEX_SPECIAL:
            out.append(_TEX_SPECIAL[ch])
        else:
            d = unicodedata.normalize("NFD", ch)
            base, marks = d[0], d[1:]
            if ord(base) < 128 and marks and all(m in _TEX_ACCENT for m in marks):
                b = {"i": "\\i", "j": "\\j"}.get(base, base)
                for m in marks:
                    b = "\\" + _TEX_ACCENT[m] + b
                out.append("{" + b + "}")
            elif ord(base) < 128:
                out.append(base)  # unknown combining mark: keep the base letter
    return "".join(out)


def clean_venue(v: str) -> str:
    """Venue strings for the MDPI style: no leading 'Proceedings of the' (the style adds 'In Proceedings of the'),
    and OpenAlex's 'arXiv (Cornell University)' as 'arXiv preprint'."""
    v = " ".join((v or "").split())
    if v.lower().startswith("arxiv (cornell university)"):
        return "arXiv preprint"
    for pre in ("Proceedings of the Proceedings of the ", "Proceedings of the ", "Proceedings of "):
        if v.startswith(pre):
            return v[len(pre):]
    return v


def bibtex_entry(rec: dict) -> str:
    et = rec.get("entry_type", "misc")
    fields = [("author", " and ".join(tex_ascii(_bib_name(a)) for a in rec.get("authors", []))),
              ("title", "{" + tex_ascii(_esc(rec["title"])) + "}"), ("year", str(rec["year"]))]
    venue = tex_ascii(_esc(clean_venue(rec.get("venue") or "")))
    if et == "article":
        fields.append(("journal", venue))
    elif et == "inproceedings":
        fields.append(("booktitle", venue))
    elif et == "book":
        fields.append(("publisher", tex_ascii(_esc(rec.get("publisher") or "")) or venue))
    else:
        fields.append(("howpublished", venue or "arXiv preprint"))
    for k in ("volume", "pages"):
        if rec.get(k):
            fields.append((k, _esc(str(rec[k]))))
    if rec.get("doi"):
        fields.append(("doi", rec["doi"]))
    if rec.get("arxiv"):
        fields.append(("eprint", rec["arxiv"]))
        fields.append(("archiveprefix", "arXiv"))
    if rec.get("url"):
        fields.append(("url", rec["url"]))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{et}{{{rec['key']},\n{body}\n}}\n"
