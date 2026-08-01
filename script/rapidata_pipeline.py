"""Rapidata pipeline: replay real preference voters as VLM personas on image *pairs*.

Every other dataset script here asks one persona to score one image. This one is
pairwise: Rapidata's 700k preference corpus (data/rapidata_700k/) shows two
AI-generated images (from DALL-E 3 / Flux / MidJourney / Stable Diffusion) side
by side and asks a crowd worker "Which image do you prefer?". Each pair collects
26 votes. We re-create each voter from the only attributes the dataset ships —
their country and language — and ask the VLM to make that person's choice, then
compare the simulated vote distribution against the real one.

What the dataset gives us (data/rapidata_700k/data/train_*.parquet):

    prompt, image1, image2 (embedded bytes), image1_path, image2_path,
    model1, model2, votes_image1, votes_image2, detailed_results

``detailed_results`` is a JSON string: ``{"votes": [{"votedFor": <an image path>,
"userDetails": {"country", "language", "userScore"}}, ...], "metadata": {...}}``.
This script emits the same shape, with a ``vlm_detailed_results`` dict built from
the model's votes, so the simulated and human ballots sit side by side
(``--export-parquet``).

Four things about this data drive the design, and each one is easy to get wrong:

1. **The humans were prompt-blind.** The paper behind the dataset (arXiv:2409.11904)
   shows annotators only the two images and the question "Which image do you
   prefer?" — explicitly *not* the generation prompt, since it "is not relevant
   for style preference". So ``--criterion preference`` (the default) does not put
   the prompt in the VLM's context either; that is what makes the comparison
   apples-to-apples. ``--criterion alignment`` *does* show it and asks Rapidata's
   alignment question instead ("Which image better reflects the caption above
   them?") — a legitimate ablation, but note its ground truth lives in Rapidata's
   separate Alignment dataset, not in these preference votes.

2. **There are no rater IDs.** ``userDetails`` carries country, language, and a
   ``userScore`` (a platform reliability number, 13% of which sit at exactly the
   0.3 floor) — no stable identifier. Two votes from ('IN', 'en') may or may not
   be the same human. So unlike para/lapis/eva, a persona here is a *stratum*
   (country x language), not an individual, and "replaying rater #7" is not a
   thing this dataset can support. The userScore is recorded and can gate votes
   (``--min-user-score``) but is deliberately kept out of the prompt: it describes
   the platform's trust in the worker, not the worker's taste.

3. **Half the corpus is anonymous.** ``userDetails`` is absent for ~51% of votes
   (341,522 / 674,543) — whole shards of it: train_0006-0012 and 0018-0021 have it
   for no vote, train_0001-0004 / 0014-0016 / 0023-0026 for every vote, and
   0005/0013/0017/0022 are mixed. Votes with no demographics cannot condition a
   persona, so they are skipped by default; ``--include-anonymous`` runs them
   under the generic system prompt instead.

4. **Position bias is a first-class confound.** A VLM shown two images has a
   well-known tendency to favour whichever it sees first, and that bias would
   masquerade as preference. Presentation order is therefore randomized per vote
   (``--order random``, the default) from a hash of (seed, pair, vote) so it
   survives resume and sharding, and ``p_choose_first`` is reported: with order
   randomized independently of content, anything far from 0.5 is bias, not taste.
   ``--order both`` judges every vote in both orders and reports how often the
   answer flips.

Decoding: per CLAUDE.md, Qwen2-VL's shipped generation_config.json pins top_k=1 /
top_p=0.001, so a temperature alone leaves decoding at argmax and every voter of a
pair returns a byte-identical answer. This script always passes --top-k/--top-p
explicitly (defaults 0 / 1.0 = pure temperature sampling) and reports
``degeneracy`` (how many pairs came back all-identical) so a collapsed run is
visible rather than silent.

Logs follow para_pipeline's chunked format and reuse its machinery: a small
summary <stem>.json (config, per-pair metadata, metrics, chunk manifest) beside
<stem>.part-NNNN.json vote chunks, so a checkpoint costs O(new votes) rather than
rewriting a 675k-vote file. --resume LOG_JSON picks a crashed run back up
(re-running the votes that failed), and --shard i/N slices the *pairs* across
processes — by pair, never by vote, since every metric here needs a pair's whole
ballot in one place. Merge the shard logs afterwards with --analyze-only.

Watch the vocabulary: a *split* is one of the 26 train_NNNN parquet files
(--splits), a *shard* is a slice of this run's task list (--shard i/N).

Run from the repo root (persona conda env):

    python script/rapidata_pipeline.py --n-pairs 20 --temperature 0.7
    python script/rapidata_pipeline.py --n-pairs 20 --persona-blind --temperature 0.7
    python script/rapidata_pipeline.py --n-pairs 200 --order both --export-parquet
    python script/rapidata_pipeline.py --resume data/logs/rapidata_<ts>.json --n-pairs 200
    # 4 GPUs over disjoint pairs, then merge:
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python script/rapidata_pipeline.py \
        --n-pairs 800 --shard $i/4 --output data/logs/rapidata_run.json & done; wait
    python script/rapidata_pipeline.py --analyze-only \
        "$(ls -m data/logs/rapidata_run.shard*of4.json | tr -d ' \n')"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The chunked-log, shard-spec and retry machinery in para_pipeline is
# dataset-agnostic (it only ever sees a summary path and a list of records), so it
# is imported rather than re-implemented — a rapidata log is readable by the same
# tools, including script/export_results.py.
from para_pipeline import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    _atomic_write_json,
    _flush_chunks,
    _n_parts,
    _parse_shard,
    _part_path,
    _pearson,
    _prune_stale_parts,
    _read_log_and_results,
    _shard_suffix,
    _spearman,
    generate_with_retry,
    is_generation_error,
)

RAPIDATA_DATA_DIR = REPO_ROOT / "data" / "rapidata_700k" / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

# --------------------------------------------------------------------------
# Voter attributes -> natural language
# --------------------------------------------------------------------------
# The persona env has no pycountry/babel, so the ISO 3166-1 alpha-2 and ISO 639-1
# codes actually present in the corpus (136 countries, 57 languages) are spelled
# out here. Anything unmapped falls back to the raw code, which keeps a stray or
# invalid code (the corpus contains 'UD' and 'unknown') from crashing a run.

_COUNTRY_NAMES = {
    "AE": "the United Arab Emirates", "AF": "Afghanistan", "AL": "Albania",
    "AM": "Armenia", "AO": "Angola", "AR": "Argentina", "AT": "Austria",
    "AU": "Australia", "AZ": "Azerbaijan", "BA": "Bosnia and Herzegovina",
    "BD": "Bangladesh", "BE": "Belgium", "BF": "Burkina Faso", "BG": "Bulgaria",
    "BJ": "Benin", "BO": "Bolivia", "BR": "Brazil", "BT": "Bhutan",
    "BW": "Botswana", "BY": "Belarus", "CA": "Canada",
    "CD": "the Democratic Republic of the Congo", "CG": "the Republic of the Congo",
    "CH": "Switzerland", "CI": "Cote d'Ivoire", "CL": "Chile", "CM": "Cameroon",
    "CN": "China", "CO": "Colombia", "CR": "Costa Rica", "CV": "Cape Verde",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "DO": "the Dominican Republic", "DZ": "Algeria", "EC": "Ecuador",
    "EE": "Estonia", "EG": "Egypt", "ES": "Spain", "ET": "Ethiopia",
    "FI": "Finland", "FR": "France", "GB": "the United Kingdom", "GE": "Georgia",
    "GH": "Ghana", "GR": "Greece", "GT": "Guatemala", "GU": "Guam",
    "HK": "Hong Kong", "HN": "Honduras", "HR": "Croatia", "HT": "Haiti",
    "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IN": "India",
    "IQ": "Iraq", "IT": "Italy", "JM": "Jamaica", "JO": "Jordan", "JP": "Japan",
    "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia", "KR": "South Korea",
    "KZ": "Kazakhstan", "LB": "Lebanon", "LC": "Saint Lucia", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya", "MA": "Morocco",
    "MD": "Moldova", "ME": "Montenegro", "ML": "Mali", "MM": "Myanmar",
    "MR": "Mauritania", "MU": "Mauritius", "MW": "Malawi", "MX": "Mexico",
    "MY": "Malaysia", "MZ": "Mozambique", "NA": "Namibia", "NG": "Nigeria",
    "NI": "Nicaragua", "NL": "the Netherlands", "NO": "Norway", "NP": "Nepal",
    "NZ": "New Zealand", "OM": "Oman", "PA": "Panama", "PE": "Peru",
    "PH": "the Philippines", "PK": "Pakistan", "PL": "Poland",
    "PR": "Puerto Rico", "PS": "Palestine", "PT": "Portugal", "PY": "Paraguay",
    "RO": "Romania", "RS": "Serbia", "RU": "Russia", "RW": "Rwanda",
    "SA": "Saudi Arabia", "SC": "Seychelles", "SD": "Sudan", "SE": "Sweden",
    "SG": "Singapore", "SI": "Slovenia", "SK": "Slovakia", "SN": "Senegal",
    "SO": "Somalia", "SV": "El Salvador", "SY": "Syria", "SZ": "Eswatini",
    "TH": "Thailand", "TN": "Tunisia", "TR": "Turkey", "TT": "Trinidad and Tobago",
    "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda", "US": "the United States",
    "UY": "Uruguay", "UZ": "Uzbekistan", "VE": "Venezuela", "VN": "Vietnam",
    "XK": "Kosovo", "YE": "Yemen", "ZA": "South Africa", "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

_LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian",
    "ca": "Catalan", "ch": "Chamorro", "co": "Corsican", "cs": "Czech",
    "da": "Danish", "de": "German", "dz": "Dzongkha", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fi": "Finnish", "fr": "French", "gu": "Gujarati",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "ka": "Georgian", "ko": "Korean", "lv": "Latvian", "ml": "Malayalam",
    "mr": "Marathi", "ms": "Malay", "nb": "Norwegian", "nl": "Dutch",
    "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sk": "Slovak", "sm": "Samoan", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "ve": "Venda", "vi": "Vietnamese", "zh": "Chinese",
}

# Codes that carry no information about the voter: 'unknown' is the corpus's own
# placeholder and 'UD' is not an assigned alpha-2 code.
_NULL_COUNTRY_CODES = {"unknown", "UD", "", None}


def country_name(code: Optional[str]) -> Optional[str]:
    """Human-readable country for an alpha-2 code, or None if it says nothing."""
    if code in _NULL_COUNTRY_CODES:
        return None
    return _COUNTRY_NAMES.get(code, str(code))


def language_name(code: Optional[str]) -> Optional[str]:
    """Human-readable language for a 639-1 code, or None if it says nothing.

    The corpus mixes case ('EN' and 'en' both occur), so codes are lowercased
    before lookup.
    """
    if not code:
        return None
    return _LANGUAGE_NAMES.get(str(code).lower(), str(code))


def _article(word: str) -> str:
    """'a' / 'an' for a language name — these land verbatim in the prompt, and
    'a English-speaking person' reads like a bug in the persona."""
    return "an" if word[:1].upper() in "AEIOU" else "a"


def build_rapidata_description(country: Optional[str], language: Optional[str]) -> str:
    """Turn one voter's (country, language) into the free-text persona description.

    Returns '' when neither attribute says anything, which is the caller's cue to
    fall back to the generic (unconditioned) system prompt.
    """
    where, speaks = country_name(country), language_name(language)
    if where and speaks:
        return f"a person from {where} who speaks {speaks}"
    if where:
        return f"a person from {where}"
    if speaks:
        return f"{_article(speaks)} {speaks}-speaking person"
    return ""


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
# Per the repo convention, dataset-specific prompts live in the dataset's script:
# Rapidata's voters are described only by country/language and the answer is a
# binary choice, so none of PARA's 1-5 scoring machinery applies here.

RAPIDATA_SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing as one specific participant in a worldwide, "
    "crowdsourced study of image preferences. The participant is {description}\n\n"
    "You are shown two AI-generated images and asked which one you prefer. "
    "Answer the way this exact person would: let their cultural background, "
    "the visual world they live in, and their own taste decide which image "
    "appeals to them more. This is personal preference, not a test — there is "
    "no correct answer, and you must pick one of the two. Do not mention that "
    "you are an AI or that you are role-playing."
)

# Persona-blind control (--persona-blind): same task, no voter conditioning, so
# the model's unconditioned preference can be contrasted with the persona runs.
RAPIDATA_GENERIC_SYSTEM_PROMPT = (
    "You are taking part in a large study of image preferences.\n\n"
    "You are shown two AI-generated images and asked which one you prefer. "
    "Judge them on their own merits, without adopting any particular person's "
    "perspective or taste. You must pick one of the two. Do not mention that "
    "you are an AI."
)

# The wording of both questions is Rapidata's own (arXiv:2409.11904). The
# preference question is the one these votes actually answer; the alignment
# question is offered for --criterion alignment.
_PREFERENCE_QUESTION = "Which image do you prefer?"
_ALIGNMENT_QUESTION = "Which image better reflects the caption above them?"


def build_rapidata_question(
    criterion: str = "preference",
    prompt_text: Optional[str] = None,
    persona_blind: bool = False,
) -> str:
    """Build the user turn asking for a choice between Image A and Image B.

    ``prompt_text`` is included only when the criterion asks about the caption —
    the humans who produced these votes never saw it (see module docstring).
    """
    if criterion == "alignment":
        if not prompt_text:
            raise ValueError("--criterion alignment needs the pair's prompt text")
        head = f'Both images were generated from this caption:\n\n"{prompt_text}"\n\n{_ALIGNMENT_QUESTION}'
    else:
        head = _PREFERENCE_QUESTION
        if prompt_text:
            head = (
                f'Both images were generated from this caption:\n\n"{prompt_text}"\n\n'
                f"{_PREFERENCE_QUESTION}"
            )
    voice = "" if persona_blind else "in-character "
    return (
        f"{head}\n\n"
        'Answer "A" for the first image or "B" for the second. You must choose '
        "one; do not answer with both or neither.\n\n"
        "Respond with ONLY a single JSON object and nothing else (no markdown, no "
        "extra text), in exactly this form:\n"
        f'{{"choice": "A" or "B", "comment": "<one short {voice}sentence saying why>"}}'
    )


def rapidata_system_prompt(description: str) -> str:
    if not description:
        return RAPIDATA_GENERIC_SYSTEM_PROMPT
    return RAPIDATA_SYSTEM_PROMPT_TEMPLATE.format(description=description)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"')
_CHOICE_RE = re.compile(r'"choice"\s*:\s*"?\s*([AB])\b', re.IGNORECASE)
# Last-ditch: a bare "A"/"B" or "Image A" somewhere in an otherwise unparseable
# reply. Anchored to avoid matching the 'a' in ordinary prose.
_LOOSE_CHOICE_RE = re.compile(r"\b(?:image\s*)?([AB])\b")


def parse_rapidata_choice(raw_response: str) -> tuple[Optional[str], str]:
    """Parse a raw response into ``("A"|"B"|None, comment)``.

    Mirrors parse_para_rating's strategy (strict JSON first, then regex over dirty
    output) but for a binary label. An unparseable or ambiguous reply yields None
    rather than raising, so one bad response doesn't abort the batch — those votes
    are dropped from every metric.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()

    start, end = text.find("{"), text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end > start else text
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        data = None

    choice = None
    if isinstance(data, dict) and "choice" in data:
        value = str(data["choice"]).strip().upper()
        if value in ("A", "B"):
            choice = value
    if choice is None:
        match = _CHOICE_RE.search(text)
        if match:
            choice = match.group(1).upper()
    if choice is None:
        # Only trust a bare letter if exactly one of A/B appears — "A or B" or a
        # reply naming both is genuinely ambiguous and must not be guessed.
        found = {m.group(1).upper() for m in _LOOSE_CHOICE_RE.finditer(text)}
        if len(found) == 1:
            choice = found.pop()

    if isinstance(data, dict) and "comment" in data:
        comment = str(data.get("comment", "")).strip()
    else:
        match = _COMMENT_RE.search(text)
        comment = match.group(1) if match else text
    return choice, comment


# --------------------------------------------------------------------------
# Loading pairs and votes
# --------------------------------------------------------------------------


def _split_name(path: Path) -> str:
    """'train_0001-00000-of-00001.parquet' -> 'train_0001'.

    Note the deliberate vocabulary split, since two unrelated things would
    otherwise both be called a shard: a *split* is one of the dataset's 26
    train_NNNN parquet files (--splits), while a *shard* is a slice of this run's
    task list handed to one process (--shard i/N), exactly as in para_pipeline.
    """
    return path.name.split("-")[0]


def resolve_splits(spec: Optional[str]) -> list[Path]:
    """Parquet files to read, from a comma-separated list of split names or globs."""
    if not spec:
        default = sorted(RAPIDATA_DATA_DIR.glob("train_0001-*.parquet"))
        if not default:
            raise SystemExit(f"no train_0001-*.parquet under {RAPIDATA_DATA_DIR}")
        return [default[0]]
    paths: list[Path] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        hits = sorted(RAPIDATA_DATA_DIR.glob(f"{token}-*.parquet")) or sorted(
            RAPIDATA_DATA_DIR.glob(token)
        )
        if not hits:
            raise SystemExit(f"no parquet matched '{token}' in {RAPIDATA_DATA_DIR}")
        paths.extend(hits)
    return sorted(set(paths))


def iter_pairs(
    splits: list[Path], columns: list[str]
) -> Iterator[tuple[str, int, dict]]:
    """Yield (split_name, row_index, row_dict) over the given parquet files."""
    import pyarrow.parquet as pq

    for path in splits:
        table = pq.read_table(path, columns=columns).to_pydict()
        name = _split_name(path)
        for i in range(len(table[columns[0]])):
            yield name, i, {c: table[c][i] for c in columns}


def _usable_votes(
    raw_votes: list[dict],
    image1_path: str,
    image2_path: str,
    include_anonymous: bool,
    min_user_score: float,
) -> list[dict]:
    """Normalize one pair's ballots into the votes we can actually replay.

    Drops votes with no ``userDetails`` (~51% of the corpus — see the module
    docstring) unless ``include_anonymous``, votes below the userScore gate, and
    votes whose ``votedFor`` names neither image of the pair.
    """
    out = []
    for idx, vote in enumerate(raw_votes):
        voted_for = vote.get("votedFor")
        if voted_for == image1_path:
            human_choice = 1
        elif voted_for == image2_path:
            human_choice = 2
        else:
            continue

        details = vote.get("userDetails")
        if not isinstance(details, dict):
            if not include_anonymous:
                continue
            details = {}
        score = details.get("userScore")
        if min_user_score > 0 and (score is None or float(score) < min_user_score):
            continue
        out.append(
            {
                "vote_idx": idx,
                "country": details.get("country"),
                "language": details.get("language"),
                "user_score": score,
                "human_choice": human_choice,
            }
        )
    return out


def _stable_seed(*parts) -> int:
    """A reproducible 64-bit seed from arbitrary parts.

    Python's builtin hash() is salted per interpreter (PYTHONHASHSEED), so using
    it here would hand --resume a *different* vote subset on every run.
    """
    key = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _shown_first(pair_id: str, vote_idx: int, replicate: int, order: str, seed: int) -> int:
    """Which dataset image (1 or 2) is presented as 'Image A' for this vote.

    Derived by hashing (seed, pair, vote, replicate) rather than by drawing from a
    stream RNG, so a vote's order is a pure function of its identity: --resume,
    --shard, and a different --n-pairs all reproduce the same assignment instead
    of silently re-rolling it.
    """
    if order == "first":
        return 1
    if order == "second":
        return 2
    if order == "both":
        return 1 if replicate == 0 else 2
    return 1 if _stable_seed(seed, pair_id, vote_idx) & 1 else 2


def build_tasks(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict]]:
    """Select pairs and expand them into one task per (vote, presentation order).

    Returns the task list and a ``pair_id -> pair metadata`` map (prompt, model
    names, image paths, and the human ballot restricted to the replayed votes).
    """
    splits = resolve_splits(args.splits)
    columns = [
        "prompt", "image1_path", "image2_path", "model1", "model2",
        "votes_image1", "votes_image2", "detailed_results",
    ]

    pairs: dict[str, dict] = {}
    for split, row_idx, row in iter_pairs(splits, columns):
        votes = _usable_votes(
            json.loads(row["detailed_results"])["votes"],
            row["image1_path"],
            row["image2_path"],
            args.include_anonymous,
            args.min_user_score,
        )
        if not votes:
            continue
        pair_id = f"{split}:{row_idx}"
        pairs[pair_id] = {
            "pair_id": pair_id, "split": split, "row": row_idx,
            "prompt": row["prompt"],
            "image1_path": row["image1_path"], "image2_path": row["image2_path"],
            "model1": row["model1"], "model2": row["model2"],
            "dataset_votes_image1": row["votes_image1"],
            "dataset_votes_image2": row["votes_image2"],
            "votes": votes,
        }

    if not pairs:
        raise SystemExit(
            "no pairs with usable votes — every vote in the selected split(s) is "
            "anonymous. Pass --include-anonymous, or pick a split that ships "
            "userDetails (train_0001-0004, 0014-0016, 0023-0026)."
        )

    pair_ids = sorted(pairs)
    if args.n_pairs and args.n_pairs < len(pair_ids):
        rng = np.random.default_rng(args.seed)
        pair_ids = sorted(rng.choice(pair_ids, size=args.n_pairs, replace=False).tolist())

    if args.shard:
        shard_i, shard_n = _parse_shard(args.shard)
        # Round-robin over pairs, never over votes: every metric that matters here
        # is per-pair (vote share, EMD-style distribution match, the bootstrap
        # floor), so a pair's whole ballot has to stay inside one process. Same
        # reason para_pipeline shards by image rather than by rating.
        pair_ids = pair_ids[shard_i::shard_n]
        print(f"Shard {shard_i}/{shard_n}: this process handles {len(pair_ids)} "
              f"of the selected pairs.")

    pairs = {p: pairs[p] for p in pair_ids}

    n_replicates = 2 if args.order == "both" else 1
    tasks: list[dict] = []
    for pair_id in pair_ids:
        pair = pairs[pair_id]
        votes = pair["votes"]
        if args.votes_per_pair and args.votes_per_pair < len(votes):
            rng = np.random.default_rng(_stable_seed(args.seed, pair_id))
            keep = sorted(rng.choice(len(votes), size=args.votes_per_pair, replace=False))
            votes = [votes[i] for i in keep]
            pair["votes"] = votes
        for vote in votes:
            for replicate in range(n_replicates):
                tasks.append(
                    {
                        "pair_id": pair_id,
                        "vote_idx": vote["vote_idx"],
                        "replicate": replicate,
                        "country": vote["country"],
                        "language": vote["language"],
                        "user_score": vote["user_score"],
                        "human_choice": vote["human_choice"],
                        "shown_first": _shown_first(
                            pair_id, vote["vote_idx"], replicate, args.order, args.seed
                        ),
                    }
                )
    return tasks, pairs


def _load_pair_images(splits: list[Path], pair_ids: set[str], image_size: Optional[int]) -> dict:
    """Decode the embedded image bytes for the selected pairs only.

    The images live inside the parquet (image1/image2 are struct<bytes, path>);
    the *_path columns are dataset identifiers like 'flux/155_0.jpg' and do not
    resolve against data/rapidata_700k/raw_data/images/, whose directories are
    named for full model versions. So bytes are the only reliable source.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    wanted_rows: dict[str, set[int]] = {}
    for pair_id in pair_ids:
        split, row = pair_id.rsplit(":", 1)
        wanted_rows.setdefault(split, set()).add(int(row))

    images: dict[str, tuple] = {}
    for path in splits:
        name = _split_name(path)
        if name not in wanted_rows:
            continue
        table = pq.read_table(path, columns=["image1", "image2"]).to_pydict()
        for row in sorted(wanted_rows[name]):
            decoded = []
            for column in ("image1", "image2"):
                image = Image.open(io.BytesIO(table[column][row]["bytes"])).convert("RGB")
                if image_size:
                    image.thumbnail((image_size, image_size), Image.LANCZOS)
                decoded.append(image)
            images[f"{name}:{row}"] = tuple(decoded)
    return images


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _share_resample_mae(human_choices, n_draw: int, rng, n_boot: int = 400) -> float:
    """Sampling-noise floor for |vlm_share - human_share| on one pair.

    Even a model that draws perfectly from the true human distribution lands
    ``n_draw`` votes away from the humans' own share by chance. This is the
    expected error of exactly that ideal sampler — the pairwise analogue of
    para_pipeline's human_resample_emd. A model at this floor is
    indistinguishable from human sampling noise; below it means nothing.
    """
    truth = np.asarray(human_choices) == 1
    share = truth.mean()
    draws = rng.choice(truth, size=(n_boot, n_draw), replace=True).mean(axis=1)
    return float(np.mean(np.abs(draws - share)))


def _pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the vote-level frame to one row per pair (human vs VLM shares)."""
    rows = []
    for pair_id, group in df.groupby("pair_id", sort=True):
        parsed = group[group["pred_choice"].notna()]
        # Under --order both each human vote appears twice (once per presentation
        # order), so the human side is de-duplicated by vote_idx: the ballot has
        # one entry per real voter no matter how many times we replayed them.
        humans = group.drop_duplicates("vote_idx")
        # human_share1 is computed over the votes we actually replayed, not the
        # dataset's full votes_image1/2, so both sides describe the same ballot.
        rows.append(
            {
                "pair_id": pair_id,
                "model1": group["model1"].iloc[0],
                "model2": group["model2"].iloc[0],
                "n_human": len(humans),
                "n_vlm": len(parsed),
                "human_share1": float((humans["human_choice"] == 1).mean()),
                "vlm_share1": (
                    float((parsed["pred_choice"] == 1).mean()) if len(parsed) else np.nan
                ),
                "n_responses": len(group),
                "n_distinct_responses": group["raw_response"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _vote_level(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    parsed = df[df["pred_choice"].notna()]
    if parsed.empty:
        return {"n_votes": int(len(df)), "n_parsed": 0, "parse_rate": 0.0}

    share = pairs.set_index("pair_id")["human_share1"]
    per_vote_share = parsed["pair_id"].map(share).to_numpy()
    # A model that samples from each pair's true human distribution matches a
    # random human vote this often; a model that always predicts the pair's
    # majority does this well. The first is the honest target for temperature
    # sampling, the second is the ceiling for any predictor.
    sampling_accuracy = float(np.mean(per_vote_share**2 + (1 - per_vote_share) ** 2))
    majority_ceiling = float(np.mean(np.maximum(per_vote_share, 1 - per_vote_share)))

    accuracy = float((parsed["pred_choice"] == parsed["human_choice"]).mean())
    p_first = float((parsed["pred_choice"] == parsed["shown_first"]).mean())
    global_share1 = float((parsed["human_choice"] == 1).mean())

    out = {
        "n_votes": int(len(df)),
        "n_parsed": int(len(parsed)),
        "parse_rate": float(len(parsed) / len(df)),
        "vote_accuracy": accuracy,
        "baseline_random": 0.5,
        # Always guessing whichever image the crowd favours overall.
        "baseline_global_majority": max(global_share1, 1 - global_share1),
        "human_sampling_accuracy": sampling_accuracy,
        "human_majority_ceiling": majority_ceiling,
        "p_choose_first": p_first,
        "p_choose_image1": float((parsed["pred_choice"] == 1).mean()),
        "human_p_image1": global_share1,
    }
    if "replicate" in df.columns and df["replicate"].nunique() > 1:
        both = parsed.pivot_table(
            index=["pair_id", "vote_idx"], columns="replicate", values="pred_choice"
        ).dropna()
        if len(both) and both.shape[1] == 2:
            # Same voter, same images, opposite presentation order. Read this
            # alongside p_choose_first, not instead of it: at temperature > 0 a
            # flip is position bias *plus* ordinary sampling noise (an unbiased
            # sampler picking image 1 with probability p still flips at 2p(1-p),
            # i.e. up to 0.5 on a genuine toss-up), so a high flip rate alone
            # proves nothing. p_choose_first is the clean bias measure, since
            # order is assigned independently of content.
            out["order_flip_rate"] = float((both[0] != both[1]).mean())
            out["n_order_replicate_pairs"] = int(len(both))
    return out


def _pair_level(pairs: pd.DataFrame, df: pd.DataFrame, rng, tie_margin: float) -> dict:
    usable = pairs[pairs["vlm_share1"].notna()]
    if usable.empty:
        return {"n_pairs": int(len(pairs)), "n_pairs_scored": 0}

    human, vlm = usable["human_share1"].to_numpy(), usable["vlm_share1"].to_numpy()
    error = np.abs(vlm - human)

    human_by_pair = {
        pair_id: group.drop_duplicates("vote_idx")["human_choice"].to_numpy()
        for pair_id, group in df.groupby("pair_id", sort=True)
    }
    floor = float(
        np.mean(
            [
                _share_resample_mae(human_by_pair[row["pair_id"]], int(row["n_vlm"]), rng)
                for _, row in usable.iterrows()
            ]
        )
    )

    decided = usable[np.abs(usable["human_share1"] - 0.5) > tie_margin]
    winner_agreement = (
        float(
            (
                (decided["human_share1"] > 0.5) == (decided["vlm_share1"] > 0.5)
            ).mean()
        )
        if len(decided)
        else float("nan")
    )
    return {
        "n_pairs": int(len(pairs)),
        "n_pairs_scored": int(len(usable)),
        "share_mae": float(error.mean()),
        "share_rmse": float(np.sqrt((error**2).mean())),
        "share_bias": float(np.mean(vlm - human)),
        "share_mae_human_floor": floor,
        # <1 would mean the model tracks the crowd more closely than a same-sized
        # sample of the crowd tracks itself, i.e. it is at the noise floor.
        "share_mae_over_floor": float(error.mean() / floor) if floor else float("nan"),
        "baseline_share_mae_constant_half": float(np.mean(np.abs(0.5 - human))),
        "baseline_share_mae_global_mean": float(np.mean(np.abs(human.mean() - human))),
        "pearson_share": _pearson(vlm, human),
        "spearman_share": _spearman(vlm, human),
        "brier": float(np.mean((vlm - human) ** 2)),
        "winner_agreement": winner_agreement,
        "n_pairs_decided": int(len(decided)),
        "tie_margin": tie_margin,
    }


def _model_level(df: pd.DataFrame) -> dict:
    """Per-generator win rates, human vs VLM — the Rapidata leaderboard view."""
    parsed = df[df["pred_choice"].notna()]
    matchups = []
    for (m1, m2), group in df.groupby(["model1", "model2"], sort=True):
        got = group[group["pred_choice"].notna()]
        matchups.append(
            {
                "model1": m1, "model2": m2,
                "n_votes": int(len(group)),
                "human_win_rate_model1": float((group["human_choice"] == 1).mean()),
                "vlm_win_rate_model1": (
                    float((got["pred_choice"] == 1).mean()) if len(got) else float("nan")
                ),
            }
        )

    per_model: dict[str, dict] = {}
    for model in sorted(set(df["model1"]) | set(df["model2"])):
        as1_h, as2_h = df[df["model1"] == model], df[df["model2"] == model]
        as1_v, as2_v = parsed[parsed["model1"] == model], parsed[parsed["model2"] == model]
        n_h = len(as1_h) + len(as2_h)
        n_v = len(as1_v) + len(as2_v)
        if not n_h:
            continue
        wins_h = int((as1_h["human_choice"] == 1).sum() + (as2_h["human_choice"] == 2).sum())
        wins_v = int((as1_v["pred_choice"] == 1).sum() + (as2_v["pred_choice"] == 2).sum())
        per_model[model] = {
            "n_votes": n_h,
            "human_win_rate": wins_h / n_h,
            "vlm_win_rate": (wins_v / n_v) if n_v else float("nan"),
        }
    return {"matchups": matchups, "per_model": per_model}


def _group_level(df: pd.DataFrame, key: str, min_votes: int = 30) -> list[dict]:
    """Per-country / per-language agreement — is conditioning buying anything?"""
    parsed = df[df["pred_choice"].notna()]
    rows = []
    for value, group in parsed.groupby(key, sort=True):
        if len(group) < min_votes:
            continue
        rows.append(
            {
                key: value,
                "n_votes": int(len(group)),
                "vote_accuracy": float((group["pred_choice"] == group["human_choice"]).mean()),
                "vlm_p_image1": float((group["pred_choice"] == 1).mean()),
                "human_p_image1": float((group["human_choice"] == 1).mean()),
                "p_choose_first": float((group["pred_choice"] == group["shown_first"]).mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["n_votes"])


def _degeneracy(pairs: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Did decoding actually sample? (See the CLAUDE.md decoding gotcha.)

    If top_k/top_p are left at Qwen2-VL's shipped values, every voter of a pair
    returns byte-identical text and the vote distribution collapses to a point —
    the run looks fine but the per-pair spread is an artifact.
    """
    multi = pairs[pairs["n_responses"] > 1]
    if multi.empty:
        return {}
    identical = (multi["n_distinct_responses"] == 1).mean()
    return {
        "mean_distinct_response_ratio": float(
            (multi["n_distinct_responses"] / multi["n_responses"]).mean()
        ),
        "frac_pairs_all_identical": float(identical),
        "mean_vlm_share_sd": float(pairs["vlm_share1"].std()),
    }


def compute_metrics(records: list[dict], seed: int = 0, tie_margin: float = 0.0) -> dict:
    df = pd.DataFrame(records)
    if df.empty:
        return {}
    # Country codes include 'NA' (Namibia); keep them as plain strings so nothing
    # downstream reads them as missing values.
    df["pred_choice"] = pd.to_numeric(df["pred_choice"], errors="coerce")
    rng = np.random.default_rng(seed)
    pairs = _pair_frame(df)
    return {
        "vote_level": _vote_level(df, pairs),
        "pair_level": _pair_level(pairs, df, rng, tie_margin),
        "model_level": _model_level(df),
        "by_country": _group_level(df, "country"),
        "by_language": _group_level(df, "language"),
        "degeneracy": _degeneracy(pairs, df),
    }


def print_summary(metrics: dict) -> None:
    vote, pair = metrics.get("vote_level", {}), metrics.get("pair_level", {})
    if not vote:
        print("no metrics (no records)")
        return

    print("\n" + "=" * 74)
    print("VOTE LEVEL — does the VLM make the choice this voter made?")
    print("=" * 74)
    print(f"  votes                        {vote['n_votes']} "
          f"({vote['n_parsed']} parsed, {vote['parse_rate']:.1%})")
    if not vote.get("n_parsed"):
        print("  nothing parsed — no agreement metrics")
        return
    print(f"  vote accuracy                {vote['vote_accuracy']:.3f}")
    print(f"    vs random                  {vote['baseline_random']:.3f}")
    print(f"    vs always-crowd-favourite  {vote['baseline_global_majority']:.3f}")
    print(f"    ideal sampler would get    {vote['human_sampling_accuracy']:.3f}"
          "   <- honest target at temperature > 0")
    print(f"    perfect majority predictor {vote['human_majority_ceiling']:.3f}"
          "   <- ceiling for any predictor")
    print(f"  P(choose first-shown image)  {vote['p_choose_first']:.3f}"
          "   <- 0.5 = no position bias")
    if "order_flip_rate" in vote:
        print(f"  answer flips when order swaps  {vote['order_flip_rate']:.3f} "
              f"(n={vote['n_order_replicate_pairs']}) <- includes sampling noise, "
              "not bias alone")

    if pair.get("n_pairs_scored"):
        print("\n" + "=" * 74)
        print("PAIR LEVEL — does the VLM reproduce the crowd's split?")
        print("=" * 74)
        print(f"  pairs                        {pair['n_pairs_scored']}/{pair['n_pairs']}")
        print(f"  vote-share MAE               {pair['share_mae']:.3f}")
        print(f"    human sampling floor       {pair['share_mae_human_floor']:.3f} "
              f"(ratio {pair['share_mae_over_floor']:.2f}; 1.0 = at the noise floor)")
        print(f"    vs constant 0.5            {pair['baseline_share_mae_constant_half']:.3f}")
        print(f"    vs predict global mean     {pair['baseline_share_mae_global_mean']:.3f}")
        print(f"  vote-share bias (vlm-human)  {pair['share_bias']:+.3f}")
        print(f"  pearson / spearman           {pair['pearson_share']:.3f} / "
              f"{pair['spearman_share']:.3f}")
        print(f"  winner agreement             {pair['winner_agreement']:.3f} "
              f"(n={pair['n_pairs_decided']}, tie margin {pair['tie_margin']})")

    deg = metrics.get("degeneracy") or {}
    if deg:
        print("\n" + "=" * 74)
        print("DECODING — did sampling actually happen?")
        print("=" * 74)
        print(f"  distinct responses per pair  {deg['mean_distinct_response_ratio']:.2f} "
              "(1.0 = every voter answered differently)")
        print(f"  pairs where all replies identical  {deg['frac_pairs_all_identical']:.1%}")
        if deg["frac_pairs_all_identical"] > 0.2:
            print("  WARNING: decoding looks greedy — check --top-k/--top-p "
                  "(see the CLAUDE.md decoding gotcha).")

    model = metrics.get("model_level") or {}
    if model.get("per_model"):
        print("\n" + "=" * 74)
        print("MODEL LEVEL — win rate per image generator")
        print("=" * 74)
        print(f"  {'model':<20} {'votes':>7} {'human':>8} {'vlm':>8} {'delta':>8}")
        for name, row in sorted(
            model["per_model"].items(), key=lambda kv: -kv[1]["human_win_rate"]
        ):
            delta = row["vlm_win_rate"] - row["human_win_rate"]
            print(f"  {name:<20} {row['n_votes']:>7} {row['human_win_rate']:>8.3f} "
                  f"{row['vlm_win_rate']:>8.3f} {delta:>+8.3f}")

    for key, label in (("by_country", "country"), ("by_language", "language")):
        rows = metrics.get(key) or []
        if not rows:
            continue
        print(f"\n  top {label} strata (>=30 votes)")
        print(f"    {label:<12} {'votes':>7} {'acc':>7} {'vlm p1':>8} {'human p1':>9}")
        for row in rows[:8]:
            print(f"    {str(row[label]):<12} {row['n_votes']:>7} "
                  f"{row['vote_accuracy']:>7.3f} {row['vlm_p_image1']:>8.3f} "
                  f"{row['human_p_image1']:>9.3f}")
    print()


# --------------------------------------------------------------------------
# Log I/O
# --------------------------------------------------------------------------


def _task_key(record: dict) -> tuple:
    return (record["pair_id"], record["vote_idx"], record.get("replicate", 0))


def _build_summary(
    args: argparse.Namespace,
    pairs: dict,
    records: list[dict],
    metrics: Optional[dict],
    chunk_size: int,
    summary_path: Path,
) -> dict:
    """The small summary log: config, per-pair metadata, metrics, and a manifest

    of the result-chunk files. The votes themselves live in the .part-NNNN.json
    files, not here, so the summary stays openable at full-corpus size (675k
    votes)."""
    return {
        "dataset": "rapidata_700k",
        "splits": args.splits or "train_0001",
        "shard": args.shard,
        "model_name": args.model_name,
        "backend": args.backend,
        "criterion": args.criterion,
        "persona_blind": args.persona_blind,
        "show_prompt": args.show_prompt or args.criterion == "alignment",
        "order": args.order,
        "include_anonymous": args.include_anonymous,
        "min_user_score": args.min_user_score,
        "image_size": args.image_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt_template": (
            RAPIDATA_GENERIC_SYSTEM_PROMPT
            if args.persona_blind
            else RAPIDATA_SYSTEM_PROMPT_TEMPLATE
        ),
        "question_example": build_rapidata_question(
            args.criterion,
            "<caption>" if (args.show_prompt or args.criterion == "alignment") else None,
            args.persona_blind,
        ),
        "n_pairs": len(pairs),
        "chunk_size": chunk_size,
        "n_ratings": len(records),
        "result_parts": [
            _part_path(summary_path, k).name
            for k in range(1, _n_parts(len(records), chunk_size) + 1)
        ],
        "pairs": {
            p: {k: v for k, v in meta.items() if k != "votes"} for p, meta in pairs.items()
        },
        "metrics": metrics,
    }


def analyze_log(log_paths: list[Path], output: Optional[Path] = None) -> None:
    """Recompute metrics from one or more logs — no GPU, no model load.

    A single log is updated in place (unless ``output`` is given). Several logs —
    e.g. the per-shard outputs of a multi-GPU run — are merged: their votes are
    concatenated (deduplicated by (pair_id, vote_idx, replicate), first log wins,
    except that a real response always beats a <generation error> placeholder) and
    their per-pair metadata unioned, then combined metrics are written out. Shards
    hold disjoint pairs, so merging is what restores the whole-run picture.
    """
    logs, per_log_results = [], []
    for path in log_paths:
        log, results = _read_log_and_results(path)
        logs.append(log)
        per_log_results.append(results)

    blind = {bool(log.get("persona_blind")) for log in logs}
    criteria = {log.get("criterion") for log in logs}
    if len(blind) > 1 or len(criteria) > 1:
        raise SystemExit(
            f"refusing to merge logs that answer different questions: "
            f"persona_blind={blind}, criterion={criteria}."
        )

    merged: dict[tuple, dict] = {}
    for results in per_log_results:
        for record in results:
            key = _task_key(record)
            prior = merged.get(key)
            if prior is None or (is_generation_error(prior) and not is_generation_error(record)):
                merged[key] = record
    records = list(merged.values())

    metrics = compute_metrics(records, seed=int(logs[0].get("seed", 0)))
    chunk_size = int(logs[0].get("chunk_size", DEFAULT_CHUNK_SIZE))
    pairs = {p: meta for log in logs for p, meta in (log.get("pairs") or {}).items()}

    if len(log_paths) == 1 and output is None:
        target, out = log_paths[0], dict(logs[0])
        out["metrics"] = metrics
        out["n_ratings"] = len(records)
        if "result_parts" not in logs[0]:  # legacy inline log: keep votes inline
            out["results"] = records
    else:
        target = output or (
            DEFAULT_LOG_DIR
            / f"rapidata_merged_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _flush_chunks(target, records, chunk_size, 0)
        _prune_stale_parts(target, _n_parts(len(records), chunk_size))
        out = dict(logs[0])
        out.pop("results", None)
        out.update(
            shard=None,  # a merge is no longer any single shard's slice
            chunk_size=chunk_size,
            n_pairs=len(pairs),
            n_ratings=len(records),
            result_parts=[
                _part_path(target, k).name
                for k in range(1, _n_parts(len(records), chunk_size) + 1)
            ],
            pairs=pairs,
            metrics=metrics,
            merged_from=[str(p) for p in log_paths],
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, out)
    source = f"{len(log_paths)} logs" if len(log_paths) > 1 else str(log_paths[0])
    print(f"Recomputed metrics from {source} -> {target} "
          f"({len(records)} votes over {len(pairs)} pairs; "
          f"persona_blind={logs[0].get('persona_blind')}, "
          f"criterion={logs[0].get('criterion')}, order={logs[0].get('order')}).")
    print_summary(metrics)


def export_parquet(log: dict, records: list[dict], path: Path) -> None:
    """Write the source dataset's shape, with the VLM's ballot alongside the humans'.

    Mirrors train_*.parquet — prompt, both image paths, both model names, and the
    dataset's own votes_image1/2 — then adds two ballots in the source's exact
    {"votes": [{"votedFor", "userDetails"}], "metadata": {...}} form:

      * human_detailed_results / human_votes_image1|2 — the real votes *we
        replayed*, which is what the VLM was scored against. This is not always
        the full dataset ballot (--votes-per-pair, --min-user-score and the
        anonymous-vote filter all narrow it), so it is emitted separately rather
        than conflated with the untouched votes_image1/2 above.
      * vlm_detailed_results / vlm_votes_image1|2 — the simulated votes, each
        carrying the voter it was conditioned on plus which image it saw first.

    Image bytes are not copied (the source carries ~250MB of them per 1000 rows);
    the paths identify the images.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_pair: dict[str, list[dict]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)

    def _ballot(votes: list[dict], metadata: dict) -> str:
        return json.dumps({"votes": votes, "metadata": metadata}, ensure_ascii=False)

    def _user_details(record: dict) -> dict:
        return {
            "country": record.get("country"),
            "language": record.get("language"),
            "userScore": record.get("user_score"),
        }

    rows = []
    for pair_id, meta in log["pairs"].items():
        records = by_pair.get(pair_id, [])
        path1, path2 = meta["image1_path"], meta["image2_path"]

        vlm_votes, vlm1, vlm2 = [], 0, 0
        for record in records:
            if record.get("pred_choice") not in (1, 2):
                continue
            vlm1 += record["pred_choice"] == 1
            vlm2 += record["pred_choice"] == 2
            vlm_votes.append(
                {
                    "votedFor": path1 if record["pred_choice"] == 1 else path2,
                    "userDetails": _user_details(record),
                    "shownFirst": path1 if record["shown_first"] == 1 else path2,
                    "comment": record.get("comment"),
                }
            )

        # --order both replays each real voter twice, so dedupe to one entry per voter.
        human_votes, human1, human2 = [], 0, 0
        for _, record in sorted({r["vote_idx"]: r for r in records}.items()):
            human1 += record["human_choice"] == 1
            human2 += record["human_choice"] == 2
            human_votes.append(
                {
                    "votedFor": path1 if record["human_choice"] == 1 else path2,
                    "userDetails": _user_details(record),
                }
            )

        rows.append(
            {
                "pair_id": pair_id,
                "prompt": meta["prompt"],
                "image1_path": path1,
                "image2_path": path2,
                "model1": meta["model1"],
                "model2": meta["model2"],
                "votes_image1": meta["dataset_votes_image1"],
                "votes_image2": meta["dataset_votes_image2"],
                "human_votes_image1": int(human1),
                "human_votes_image2": int(human2),
                "human_detailed_results": _ballot(
                    human_votes, {"source": "rapidata_700k", "resultType": "CompareResult"}
                ),
                "vlm_votes_image1": int(vlm1),
                "vlm_votes_image2": int(vlm2),
                "vlm_detailed_results": _ballot(
                    vlm_votes,
                    {
                        "model": log["model_name"],
                        "criterion": log["criterion"],
                        "personaBlind": log["persona_blind"],
                        "temperature": log["temperature"],
                        "order": log["order"],
                        "resultType": "CompareResult",
                        "createdAt": log["timestamp"],
                    },
                ),
            }
        )
    if not rows:
        print(f"nothing to export to {path}")
        return
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"wrote {path} ({len(rows)} pairs)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Rapidata preference voters as VLM personas conditioned on "
            "country/language, and compare simulated vs real pairwise votes."
        )
    )
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="Number of image pairs to sample (default: %(default)s).")
    parser.add_argument("--splits", default=None,
                        help="Comma-separated dataset splits (parquet files) to read, e.g. "
                             "'train_0001,train_0002' (default: train_0001). Note "
                             "train_0006-0012/0018-0021 carry no userDetails and need "
                             "--include-anonymous. Not to be confused with --shard.")
    parser.add_argument("--shard", default=None, metavar="i/N",
                        help="Run only pair shard i of N (0-based), so N processes — each "
                             "pinned to a GPU with CUDA_VISIBLE_DEVICES — cover disjoint "
                             "pairs in parallel. Sharding is by pair (all of a pair's votes "
                             "stay together, since every metric here is per-pair). Each "
                             "shard's log gets a .shardIofN suffix; merge them afterwards "
                             "with --analyze-only log1,log2,...")
    parser.add_argument("--votes-per-pair", type=int, default=None,
                        help="Cap on votes replayed per pair (default: all, ~26 — needed "
                             "for the distribution to mean anything).")
    parser.add_argument("--criterion", choices=["preference", "alignment"], default="preference",
                        help="'preference' asks Rapidata's real question with no caption "
                             "shown, matching what these voters answered; 'alignment' shows "
                             "the caption and asks which image reflects it (default: %(default)s).")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Show the generation caption in preference mode too. The humans "
                             "did NOT see it, so this is an ablation, not a fidelity fix.")
    parser.add_argument("--persona-blind", action="store_true",
                        help="Control: drop country/language conditioning and judge with one "
                             "generic prompt.")
    parser.add_argument("--order", choices=["random", "first", "second", "both"],
                        default="random",
                        help="Presentation order: 'random' per-vote coin flip (default), "
                             "'first'/'second' pin a dataset image to slot A, 'both' runs "
                             "every vote in both orders to measure position bias (2x cost).")
    parser.add_argument("--include-anonymous", action="store_true",
                        help="Also replay votes with no userDetails (~51%% of the corpus), "
                             "using the generic persona.")
    parser.add_argument("--min-user-score", type=float, default=0.0,
                        help="Drop votes whose Rapidata userScore is below this "
                             "(default: %(default)s = keep all).")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Longest side each image is resized to before it reaches the "
                             "model; 0 keeps the native 1024px (default: %(default)s). Two "
                             "1024px images cost ~2700 vision tokens per vote.")
    parser.add_argument("--tie-margin", type=float, default=0.0,
                        help="Exclude pairs whose human share is within this of 0.5 from "
                             "winner agreement (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling/order seed (default: 0).")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HF model id for the backend (default: %(default)s).")
    parser.add_argument("--backend", choices=["qwen"], default="qwen",
                        help="VLM backend. Only qwen: llava-1.5's template has one image slot "
                             "and cannot show a pair (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Votes per generate() call (default: %(default)s).")
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="Generation length; the answer is a tiny JSON object "
                             "(default: %(default)s).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: %(default)s). Must be > 0: at 0 "
                             "every voter of a pair returns the same answer and the vote "
                             "distribution collapses to a point.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="top_k (default: %(default)s = disabled). Passed explicitly "
                             "because Qwen2-VL ships top_k=1, which forces greedy decoding "
                             "whatever the temperature.")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="top_p (default: %(default)s = disabled). Passed explicitly "
                             "because Qwen2-VL ships top_p=0.001.")
    parser.add_argument("--output", default=None,
                        help="Log path (default: data/logs/rapidata_<timestamp>.json).")
    parser.add_argument("--export-parquet", nargs="?", const="", default=None,
                        help="Also write a parquet mirroring the source schema with the VLM "
                             "ballot (default path: <output stem>.parquet).")
    parser.add_argument("--resume", default=None, metavar="LOG_JSON",
                        help="Continue a previous (possibly crashed/partial) run: load the "
                             "votes already in this log, skip their tasks, and keep appending "
                             "to the same file. Must be run with the same selection args "
                             "(--n-pairs/--splits/--shard/--seed/--votes-per-pair/--order) as "
                             "the original, since those are what make the task list "
                             "reproducible.")
    parser.add_argument("--checkpoint-interval", type=float, default=300.0,
                        help="Seconds between log checkpoints (default: %(default)s).")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Max votes per result-chunk file (<stem>.part-NNNN.json); the "
                             "summary <stem>.json stays small and holds config + metrics "
                             "(default: %(default)s).")
    parser.add_argument("--analyze-only", default=None, metavar="LOG_JSON[,LOG_JSON...]",
                        help="Skip the model entirely: recompute metrics from an existing log "
                             "and print the summary. A single log is updated in place; several "
                             "comma-separated logs (e.g. the per-shard outputs of a multi-GPU "
                             "run) are merged into combined metrics.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.temperature <= 0:
        raise SystemExit(
            "--temperature must be > 0: this design gets its per-pair spread from "
            "sampling across a pair's voters, and greedy decoding makes every voter "
            "of a pair return an identical answer."
        )

    tasks, pairs = build_tasks(args)
    print(f"{len(pairs)} pairs, {len(tasks)} votes to run "
          f"(persona_blind={args.persona_blind}, criterion={args.criterion}, "
          f"order={args.order}, temperature={args.temperature})")

    records: list[dict] = []
    prior_chunked = False
    if args.resume:
        output = Path(args.resume)
        prior_log, records = _read_log_and_results(output)
        # These three change what the model was *asked*, so mixing them in one log
        # would silently pool answers to different questions.
        for field, mine in (
            ("persona_blind", bool(args.persona_blind)),
            ("criterion", args.criterion),
            ("order", args.order),
        ):
            theirs = prior_log.get(field)
            if field == "persona_blind":
                theirs = bool(theirs)
            if theirs != mine:
                raise SystemExit(
                    f"--resume log {output} has {field}={theirs!r}, which doesn't match "
                    f"this run's {field}={mine!r}; resume with matching args."
                )
        chunk_size = int(prior_log.get("chunk_size", args.chunk_size))
        # A <generation error> placeholder is a failed vote, not a completed one:
        # drop it so its task is re-run rather than skipped forever.
        prior_chunked = "result_parts" in prior_log
        n_failed = sum(1 for r in records if is_generation_error(r))
        if n_failed:
            records = [r for r in records if not is_generation_error(r)]
            if prior_chunked:
                # Dropping records shifts every later record's chunk index, so the
                # parts must be rewritten now rather than at the first checkpoint.
                _flush_chunks(output, records, chunk_size, 0)
                _prune_stale_parts(output, _n_parts(len(records), chunk_size))
        done = {_task_key(r) for r in records}
        before = len(tasks)
        tasks = [t for t in tasks if _task_key(t) not in done]
        print(f"Resuming from {output}: {before - len(tasks)} votes already done, "
              f"{len(tasks)} remaining.")
        if n_failed:
            print(f"  ({n_failed} failed votes dropped and queued for re-run.)")
    else:
        chunk_size = args.chunk_size
        # A per-shard suffix keeps parallel processes from clobbering one file.
        suffix = _shard_suffix(args.shard)
        if args.output:
            base = Path(args.output)
            output = base.with_name(base.stem + suffix + base.suffix)
        else:
            output = (
                DEFAULT_LOG_DIR
                / f"rapidata_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}{suffix}.json"
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    if not tasks:
        print("Nothing left to do.")
        metrics = compute_metrics(records, args.seed, args.tie_margin)
        # Result chunks already exist on disk (resume); just (re)write the summary.
        _atomic_write_json(
            output, _build_summary(args, pairs, records, metrics, chunk_size, output)
        )
        print_summary(metrics)
        return

    images = _load_pair_images(
        resolve_splits(args.splits),
        {t["pair_id"] for t in tasks},
        args.image_size or None,
    )

    from persona.backend import QwenVLBackend

    backend = QwenVLBackend(model_name=args.model_name, max_new_tokens=args.max_new_tokens)
    gen_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    total = len(tasks)
    started = time.time()
    last_checkpoint = time.time()
    # Votes loaded from an already-chunked resume are on disk; a legacy inline
    # resume (flushed=0) gets rewritten into chunks on the first flush.
    flushed = len(records) if prior_chunked else 0

    def _checkpoint(metrics: Optional[dict]) -> int:
        # Flush only the tail chunk(s) touched since the last checkpoint, then the
        # summary — cheap even when the run holds hundreds of thousands of votes.
        n = _flush_chunks(output, records, chunk_size, flushed)
        _atomic_write_json(
            output, _build_summary(args, pairs, records, metrics, chunk_size, output)
        )
        return n

    try:
        for start in range(0, total, args.batch_size):
            batch = tasks[start : start + args.batch_size]
            system_prompts, batch_images, prompts = [], [], []
            for task in batch:
                meta = pairs[task["pair_id"]]
                description = (
                    "" if args.persona_blind
                    else build_rapidata_description(task["country"], task["language"])
                )
                system_prompts.append(rapidata_system_prompt(description))
                image1, image2 = images[task["pair_id"]]
                # shown_first says which *dataset* image occupies slot A.
                batch_images.append(
                    [image1, image2] if task["shown_first"] == 1 else [image2, image1]
                )
                show_caption = args.show_prompt or args.criterion == "alignment"
                prompts.append(
                    build_rapidata_question(
                        args.criterion,
                        meta["prompt"] if show_caption else None,
                        args.persona_blind,
                    )
                )

            responses = generate_with_retry(
                backend, system_prompts, batch_images, prompts, gen_kwargs
            )

            for task, response in zip(batch, responses):
                meta = pairs[task["pair_id"]]
                if is_generation_error(response):
                    choice, comment, pred = None, "", None
                else:
                    choice, comment = parse_rapidata_choice(response)
                    # Slot A/B -> dataset image 1/2, undoing the order randomization.
                    if choice is None:
                        pred = None
                    elif choice == "A":
                        pred = task["shown_first"]
                    else:
                        pred = 2 if task["shown_first"] == 1 else 1
                records.append(
                    {
                        **task,
                        "split": meta["split"], "row": meta["row"],
                        "image1_path": meta["image1_path"],
                        "image2_path": meta["image2_path"],
                        "model1": meta["model1"], "model2": meta["model2"],
                        "pred_letter": choice,
                        "pred_choice": pred,
                        "comment": comment,
                        "raw_response": response,
                    }
                )

            done = start + len(batch)
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            eta = (total - done) / rate / 60 if rate else float("nan")
            print(f"  {done}/{total} votes ({rate:.2f}/s, ETA {eta:.1f} min)", flush=True)
            if time.time() - last_checkpoint >= args.checkpoint_interval:
                flushed = _checkpoint(None)
                last_checkpoint = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted after {done}/{total} votes this run — writing partial log.")

    flushed = _flush_chunks(output, records, chunk_size, flushed)
    _prune_stale_parts(output, _n_parts(len(records), chunk_size))
    metrics = compute_metrics(records, args.seed, args.tie_margin)
    log = _build_summary(args, pairs, records, metrics, chunk_size, output)
    _atomic_write_json(output, log)
    print(f"\nwrote {output} ({len(records)} votes in "
          f"{_n_parts(len(records), chunk_size)} part file(s))")
    print_summary(metrics)

    if args.export_parquet is not None:
        export_parquet(
            log,
            records,
            Path(args.export_parquet) if args.export_parquet
            else output.with_suffix(".parquet"),
        )


def main() -> None:
    args = _parse_args()
    if args.analyze_only:
        paths = [Path(p.strip()) for p in args.analyze_only.split(",") if p.strip()]
        analyze_log(paths, Path(args.output) if args.output else None)
        return
    run(args)


if __name__ == "__main__":
    main()
