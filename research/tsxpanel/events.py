"""Membership events from consecutive holdings files: names leaving / entering XIC and XCS.

Identity key is `ysym`. For each ETF the panel's present dates are walked in order; a name in file a but not in the next
file b is a deletion with bracket (last_in = a, first_out = b). Guards against fake events:
  * persistence: a deletion must stay absent from the next PERSIST present files, an addition must stay in them and
    must not have been in the previous PERSIST files (else transient: dropped from the counts); with fewer than
    PERSIST later files the event is kept and flagged `unconfirmed`;
  * ticker / name change: the leaving name's normalised name appears in b (either ETF) under another ysym ->
    `ticker_change` (symmetric for additions);
  * the identifier-scheme switch (2012-05-31) is handled upstream (panel dedupe + ysym keys); the bracket spanning it
    is flagged so its count can be inspected.
No price or return is read here. Classes: deletions `to_xcs` | `to_xic` | `ticker_change` | `left_both`;
additions `from_xcs` | `from_xic` | `ticker_change` | `new`. (Acquisitions are NOT inferred: takeover needs a logged
source, see taxloss takeover_sources.csv.)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

import _path  # noqa: F401
import placeholders

PERSIST = 2
NARROW_TDAYS = 4                      # a bracket this short or shorter has (near) daily resolution
GAP_2017 = (dt.date(2017, 1, 1), dt.date(2017, 6, 30))
OTHER = {"XIC": "XCS", "XCS": "XIC"}
ACCEPT_OFFSETS = (0, 1, 2)           # first-out file at T, T+1 or T+2 trading days: a review deletion (fund lag up to 2 days)
OTHER_FILE_MAX_DAYS = 7              # the sibling ETF's file used for the move test must be within this many days of b
EVENT_COLUMNS = ["etf", "direction", "ticker", "ysym", "name", "last_in", "first_out", "class", "weight_ref",
                 "bracket_days", "bracket_tdays", "source", "review_date", "review_offset", "n_review_candidates",
                 "review_aligned", "flags"]


def _tdays_between(cal_tday_index: dict, a, b) -> int:
    """Trading days from a to b (b later) using the calendar index; falls back to weekdays if a date is not a
    trading day in the calendar."""
    if a in cal_tday_index and b in cal_tday_index:
        return cal_tday_index[b] - cal_tday_index[a]
    n, d = 0, a
    while d < b:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _nearest_file(dates: list, target, direction: str, max_days: int):
    """The first date in sorted `dates` on/after target (direction 'after') or on/before it ('before'), if within
    max_days; else None."""
    if direction == "after":
        c = [d for d in dates if d >= target]
        d = c[0] if c else None
    else:
        c = [d for d in dates if d <= target]
        d = c[-1] if c else None
    return d if d is not None and abs((d - target).days) <= max_days else None


def build_events(panel: pd.DataFrame, cal, review_dates: list, persist: int = PERSIST) -> pd.DataFrame:
    tday_index = {d: i for i, d in enumerate(cal.tdays)}
    by = {k: g for k, g in panel.groupby(["etf", "date"])}
    dates = {e: sorted(set(panel.loc[panel["etf"] == e, "date"])) for e in panel["etf"].unique()}
    nn = panel.assign(_n=panel["name"].map(placeholders.norm_name))
    names_at = {k: dict(zip(g["_n"], g["ysym"])) for k, g in nn.groupby(["etf", "date"])}
    rows = []
    reviews = sorted(review_dates)
    for etf, ds in dates.items():
        other = OTHER.get(etf)
        for i in range(len(ds) - 1):
            a, b = ds[i], ds[i + 1]
            ga, gb = by[(etf, a)], by[(etf, b)]
            sa, sb = set(ga["ysym"]), set(gb["ysym"])
            later = [by[(etf, d)] for d in ds[i + 2:i + 2 + persist]]
            later_sets = [set(g["ysym"]) for g in later]
            earlier_sets = [set(by[(etf, d)]["ysym"]) for d in ds[max(0, i - persist):i]]
            od_b = _nearest_file(dates.get(other, []), b, "after", OTHER_FILE_MAX_DAYS)
            od_a = _nearest_file(dates.get(other, []), a, "before", OTHER_FILE_MAX_DAYS)
            ob = set(by[(other, od_b)]["ysym"]) if od_b else set()
            oa = set(by[(other, od_a)]["ysym"]) if od_a else set()
            nb = dict(names_at[(etf, b)])
            if od_b:
                nb.update({k: v for k, v in names_at[(other, od_b)].items() if k not in nb})
            na = dict(names_at[(etf, a)])
            if od_a:
                na.update({k: v for k, v in names_at[(other, od_a)].items() if k not in na})
            tdays = _tdays_between(tday_index, a, b)
            for direction, ids, ref, src in (("delete", sa - sb, ga, "a"), ("add", sb - sa, gb, "b")):
                for r in ref[ref["ysym"].isin(ids)].itertuples(index=False):
                    flags = []
                    if direction == "delete":
                        reappear = any(r.ysym in s for s in later_sets)
                        confirmed = len(later_sets) >= persist
                    else:
                        reappear = any(r.ysym not in s for s in later_sets)      # an addition must stay in
                        confirmed = len(later_sets) >= persist
                    if direction == "add" and any(r.ysym in s for s in earlier_sets):
                        continue                                                  # re-entry after a gap: transient
                    if reappear:
                        continue                                                  # transient: not an event
                    if not confirmed:
                        flags.append("unconfirmed")
                    key = placeholders.norm_name(r.name)
                    if direction == "delete":
                        if nb.get(key, r.ysym) != r.ysym:
                            cls = "ticker_change"
                        elif r.ysym in ob:
                            cls = "to_" + other.lower()
                        else:
                            cls = "left_both"
                    else:
                        if na.get(key, r.ysym) != r.ysym:
                            cls = "ticker_change"
                        elif r.ysym in oa:
                            cls = "from_" + other.lower()
                        else:
                            cls = "new"
                    if a < ishares_switch() <= b:
                        flags.append("scheme_switch_bracket")
                    if tdays > NARROW_TDAYS:
                        if a <= GAP_2017[1] and b >= GAP_2017[0]:
                            flags.append("spans_2017_gap")
                        flags.append("wide_bracket")
                    cands = [t for t in reviews if a <= t <= b]
                    cands += [t for t in reviews if t not in cands and t < b and t in tday_index and b in tday_index
                              and 0 < tday_index[b] - tday_index[t] <= max(ACCEPT_OFFSETS)]        # lagged implementation
                    near = min(cands, key=lambda t: abs((t - b).days)) if cands else None
                    off = _tdays_between(tday_index, near, b) if near is not None else None
                    aligned = tdays <= NARROW_TDAYS and off in ACCEPT_OFFSETS
                    rows.append({"etf": etf, "direction": direction, "ticker": r.ticker, "ysym": r.ysym,
                                 "name": r.name, "last_in": a, "first_out": b, "class": cls,
                                 "weight_ref": r.weight, "bracket_days": (b - a).days, "bracket_tdays": tdays,
                                 "source": "daily" if tdays <= NARROW_TDAYS else "month_end",
                                 "review_date": near, "review_offset": off, "n_review_candidates": len(cands),
                                 "review_aligned": bool(aligned), "flags": "|".join(flags)})
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev = _flag_rename_candidates(ev)
    return ev.sort_values(["first_out", "etf", "direction", "ysym"]).reset_index(drop=True)


def _first_token(name: str) -> str:
    t = placeholders.norm_name(name).split()
    return t[0] if t and len(t[0]) >= 4 else ""


def _flag_rename_candidates(ev: pd.DataFrame) -> pd.DataFrame:
    """A left_both deletion and a `new` addition in the same ETF and bracket that share the first normalised name
    token are probably one company renamed (name AND ticker changed): flag both `rename_candidate` for manual review."""
    if ev.empty:
        return ev
    ev = ev.copy()
    tok = ev["name"].map(_first_token)
    for (etf, a, b), g in ev.groupby(["etf", "last_in", "first_out"]):
        d = g[(g["direction"] == "delete") & (g["class"] == "left_both")]
        n = g[(g["direction"] == "add") & (g["class"] == "new")]
        common = set(tok[d.index]) & set(tok[n.index]) - {""}
        for i in list(d.index) + list(n.index):
            if tok[i] in common:
                ev.at[i, "flags"] = "|".join(f for f in (ev.at[i, "flags"], "rename_candidate") if f)
    return ev


def ishares_switch() -> dt.date:
    import ishares
    return ishares.DUAL_SCHEME_END


def counts_per_year(ev: pd.DataFrame) -> pd.DataFrame:
    """Events by calendar year of first_out, ETF, direction and class (wide table, plus a total column)."""
    e = ev.assign(year=ev["first_out"].map(lambda d: d.year))
    t = e.pivot_table(index=["etf", "direction", "class"], columns="year", values="ysym", aggfunc="count",
                      fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.reset_index()


def review_offset_distribution(ev: pd.DataFrame, direction="delete") -> pd.DataFrame:
    """Distribution of (first_out - nearest review date T) in trading days, narrow (daily) brackets only, renames excluded. Tells where
    a review deletion first disappears from the file relative to the Friday T."""
    n = ev[(ev["direction"] == direction) & (ev["class"] != "ticker_change") & (ev["source"] == "daily")
           & ev["review_offset"].notna()]
    return n.groupby(["etf", "review_offset"]).size().rename("n").reset_index()


def off_cycle(ev: pd.DataFrame, direction="delete") -> pd.DataFrame:
    """Off-cycle counts per year and ETF. `certain`: no review date inside the bracket (at any resolution).
    `narrow_not_review`: daily-resolution bracket whose first_out is not at offset 0..2 trading days after a review T
    (ACCEPT_OFFSETS; see review_offset_distribution)."""
    e = ev[(ev["direction"] == direction) & (ev["class"] != "ticker_change")].copy()
    e["year"] = e["first_out"].map(lambda d: d.year)
    e["certain_off_cycle"] = e["n_review_candidates"] == 0
    e["narrow_not_review"] = (e["source"] == "daily") & ~e["review_aligned"]
    g = e.groupby(["etf", "year"]).agg(events=("ysym", "count"), certain_off_cycle=("certain_off_cycle", "sum"),
                                       narrow_not_review=("narrow_not_review", "sum"),
                                       month_end_resolution=("source", lambda s: int((s == "month_end").sum())))
    return g.reset_index()
