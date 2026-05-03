"""
suspicion/module.py

SuspicionModule — pluggable behavioural analysis component.

Hierarchy:
  SuspicionModule
    ├── DetectorState   (per-player, from detectors.py)
    ├── observe_talks()  — ingest one day's talk list
    ├── observe_votes()  — ingest the actual vote map for the day
    ├── observe_execution()     — update D5 for revealed roles
    ├── observe_divine_claim()  — record claimed divine results (D5)
    ├── observe_actual_divine() — record real seer result (D5 baseline)
    ├── end_of_day_update()     — recompute σ(p) with asymmetric update rule
    ├── get_sigma(pid)   → float
    ├── get_detectors(pid) → dict of {d1..d5, E}
    └── snapshot()       → dict of full state (for suspicion trace logging)

Asymmetric σ update (spec §2):
  E(p) > 0.5 → σ(p) ← min(1.0, σ(p) + 0.15 * E(p))
  E(p) ≤ 0.5 → σ(p) ← max(0.0, σ(p) − 0.05 * (1 − E(p)))

All weight parameters (a1-a5) are passed in at construction and never
hardcoded here, in compliance with the "must be configurable" spec requirement.
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional

_DIAG = False  # set True to enable diagnostic prints

from .detectors import (
    DetectorState,
    compute_d1, compute_d2, compute_d3, compute_d4, compute_d5, compute_E,
)


_VOTE_RE      = re.compile(r"\bVOTE\s+P(\d+)", re.IGNORECASE)
_ESTIMATE_RE  = re.compile(r"\bESTIMATE\s+P(\d+)\s+WEREWOLF", re.IGNORECASE)
_DIVINED_RE   = re.compile(r"\bDIVINED\s+P(\d+)\s+(WEREWOLF|HUMAN)", re.IGNORECASE)
_COMINGOUT_RE = re.compile(r"\bCOMINGOUT\s+P(\d+)\s+(\w+)", re.IGNORECASE)
# New v2 tokens
_ATTACK_RE    = re.compile(r"\bATTACK\s+P(\d+)", re.IGNORECASE)
_DEFEND_RE    = re.compile(r"\bDEFEND\s+P(\d+)", re.IGNORECASE)
_AGREE_RE     = re.compile(r"\bAGREE\s+P(\d+)", re.IGNORECASE)
_DISAGREE_RE  = re.compile(r"\bDISAGREE\s+P(\d+)", re.IGNORECASE)
_INQUIRE_RE   = re.compile(r"\bINQUIRE\s+P(\d+)", re.IGNORECASE)
_NOT_RE       = re.compile(r"\bNOT\s+P(\d+)\s+(\w+)", re.IGNORECASE)

DEFAULT_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
}


class SuspicionModule:
    """
    Modular suspicion scorer.  Attach to any compatible agent.

    Parameters
    ----------
    player_ids      : list[int]  — all player IDs (1-indexed)
    self_id         : int        — the observing agent's own player ID
    detector_weights: dict       — a1..a5 (default: DEFAULT_WEIGHTS)
    n_wolves        : int        — number of werewolves in the game
                                   (used to detect single-wolf games for D4)
    """

    def __init__(
        self,
        player_ids:       list,
        self_id:          int,
        detector_weights: dict = None,
        n_wolves:         int  = 2,
    ):
        self._self_id   = self_id
        self._weights   = dict(DEFAULT_WEIGHTS)
        if detector_weights:
            self._weights.update(detector_weights)

        self._single_wolf = (n_wolves == 1)

        self._sigma:  Dict[int, float] = {p: 0.5 for p in player_ids}
        self._states: Dict[int, DetectorState] = {
            p: DetectorState() for p in player_ids
        }

        # Per-day accumulators (reset each day)
        self._day_talks:    list = []
        self._seen_talk_count: int = 0  # # talks already processed from current day
        self._seen_talk_day:   int = 0  # day when the above count was accumulated

        # Claimed seer divine results this game: {claimer: [(target, result)]}
        self._seer_claims: Dict[int, list] = defaultdict(list)

        # Real seer results (known from game_info.divine_result):
        self._true_divines: dict = {}

        # Track current day for context
        self._current_day: int = 1

        # For D3: store per-round suspicion pressure series
        self._round_suspicion: Dict[int, List[float]] = defaultdict(list)

        # D2: AGREE tokens accumulated this voting round {pid: [(day, agreed_pid, text)]}
        self._day_agree_tokens: Dict[int, list] = {}

        # D3: per-player per-day INQUIRE+ATTACK pressure on that player {pid: {day: count}}
        self._day_pressure: Dict[int, Dict[int, int]] = {}

        # D2: last ATTACK target per player (for checking if AGREE is on majority)
        self._last_attack_target: Dict[int, int] = {}

        # D5: claimed COMINGOUT roles {pid: role_str}
        self._comingout_claims: Dict[int, str] = {}

        # Evolution log: rows appended each end_of_day_update
        self._evolution_rows: list = []

    # ------------------------------------------------------------------ #
    #  Observation interface                                               #
    # ------------------------------------------------------------------ #

    def observe_talks(self, day: int, talk_list: list) -> None:
        """Ingest new talks. Updates D1 intended-attack, D2 agree, D3 pressure,
        D4 defenses, and D5 claim tracking."""
        self._current_day = day
        # Reset counter when a new day starts (talk_list resets each day)
        if day != self._seen_talk_day:
            self._seen_talk_count = 0
            self._seen_talk_day   = day
        new_talks = talk_list[self._seen_talk_count:]
        self._seen_talk_count = len(talk_list)
        for talk in new_talks:
            pid  = talk.agent
            text = talk.text

            # ── D1: record ATTACK token as intended accusation target ───────
            for m in _ATTACK_RE.finditer(text):
                if pid != self._self_id and pid in self._states:
                    self._states[pid].intended_attack = int(m.group(1))
                    self._last_attack_target[pid] = int(m.group(1))

            # ── D2: track AGREE tokens ──────────────────────────────────────
            for m in _AGREE_RE.finditer(text):
                if pid != self._self_id and pid in self._states:
                    agreed_with = int(m.group(1))
                    # store for post-vote analysis
                    self._day_agree_tokens.setdefault(pid, []).append(
                        (day, agreed_with, text)
                    )

            # ── D3: accumulate per-day ATTACK counts by this player ─────────
            for m in _ATTACK_RE.finditer(text):
                if pid in self._states:
                    self._states[pid].d3_round_attacks[day] = (
                        self._states[pid].d3_round_attacks.get(day, 0) + 1
                    )

            # ── D3: accumulate INQUIRE-on-p pressure ───────────────────────
            for m in _INQUIRE_RE.finditer(text):
                target = int(m.group(1))
                if target != self._self_id and target in self._states:
                    self._day_pressure.setdefault(target, {}).setdefault(day, 0)
                    self._day_pressure[target][day] = (
                        self._day_pressure[target].get(day, 0) + 1
                    )

            # ── D3: ATTACK-on-p also adds to pressure ──────────────────────
            for m in _ATTACK_RE.finditer(text):
                if pid != self._self_id:  # others attacking p
                    target = int(m.group(1))
                    if target != self._self_id and target in self._states:
                        self._day_pressure.setdefault(target, {}).setdefault(day, 0)
                        self._day_pressure[target][day] = (
                            self._day_pressure[target].get(day, 0) + 1
                        )

            # ── D4: DEFEND token ───────────────────────────────────────────
            for m in _DEFEND_RE.finditer(text):
                if pid != self._self_id and pid in self._states:
                    defended = int(m.group(1))
                    if defended != pid:
                        self._states[pid].d4_defends[defended] = (
                            self._states[pid].d4_defends.get(defended, 0) + 1
                        )
                        self._states[pid].d4_evidence.append(
                            (day, f"DEFEND P{defended}", defended)
                        )

            # ── D4: NOT Pn WEREWOLF — defending that player ────────────────
            for m in _NOT_RE.finditer(text):
                target   = int(m.group(1))
                role_str = m.group(2).upper()
                if role_str == "WEREWOLF" and pid != self._self_id and pid in self._states:
                    if target != pid:
                        self._states[pid].d4_defends[target] = (
                            self._states[pid].d4_defends.get(target, 0) + 1
                        )
                        self._states[pid].d4_evidence.append(
                            (day, f"NOT P{target} WEREWOLF", target)
                        )

            # ── D5: COMINGOUT claim ─────────────────────────────────────────
            for m in _COMINGOUT_RE.finditer(text):
                subject_pid = int(m.group(1))
                role_str    = m.group(2).upper()
                if subject_pid == pid and pid != self._self_id and pid in self._states:
                    self._states[pid].d5_total_claims += 1
                    self._comingout_claims[pid] = role_str  # track claimed role

            # ── D5: DIVINED claim ───────────────────────────────────────────
            for m in _DIVINED_RE.finditer(text):
                target = int(m.group(1))
                result = m.group(2).upper()
                if pid != self._self_id:
                    self._seer_claims[pid].append((target, result))

        # ── DIAGNOSTIC: report newly seen token types ────────────────────
        if _DIAG:
            token_counts: Counter = Counter()
            for t in new_talks:
                first_word = t.text.strip().split()[0].upper() if t.text.strip() else "EMPTY"
                token_counts[first_word] += 1
            if new_talks:
                print(f"[DIAG observe_talks] day={day} self=P{self._self_id}  "
                      f"new={len(new_talks)}  total={self._seen_talk_count}  "
                      f"token_counts={dict(token_counts)}")
                atk = {p: st.intended_attack for p, st in self._states.items()
                       if st.intended_attack is not None}
                if atk:
                    print(f"             intended_attack set: {atk}")

    def observe_votes(self, day: int, vote_map: dict) -> None:
        """
        Ingest the final vote map {voter_id: target_id} for a day.
        Updates D1 (ATTACK-mismatch), D2 (AGREE-bandwagon), D3 (pressure/aggression),
        D4 (round counter).
        """
        if _DIAG:
            import traceback as _tb
            atk = {p: st.intended_attack for p, st in self._states.items()
                   if st.intended_attack is not None}
            print(f"\n[DIAG observe_votes] *** CALLED *** day={day}  "
                  f"vote_map={vote_map}  intended_attacks_still_set={atk}")
            # Show 2-line call stack so we know WHO is calling this
            caller = _tb.extract_stack()[-2]
            print(f"             caller: {caller.filename.split('/')[-1].split(chr(92))[-1]}:"
                  f"{caller.lineno} in {caller.name}")
        from collections import Counter
        counts = Counter(vote_map.values())
        if not counts:
            return
        max_votes    = max(counts.values())
        majority_set = {t for t, c in counts.items() if c == max_votes}

        for pid, actual_target in vote_map.items():
            if pid == self._self_id or pid not in self._states:
                continue
            state = self._states[pid]

            # D1: ATTACK target vs actual vote
            intended = state.intended_attack
            if intended is not None:
                state.d1_rounds += 1
                if intended != actual_target:
                    state.d1_mismatches += 1
                    state.d1_evidence.append(
                        (day, f"ATTACK P{intended} but voted P{actual_target}")
                    )
            state.intended_attack = None
            state.intended_vote   = None   # clear legacy field too

            # D4: increment round count
            state.d4_rounds += 1

        # D2: AGREE tokens — check if the agreed-with ATTACK was on majority target
        for pid, agree_list in self._day_agree_tokens.items():
            if pid == self._self_id or pid not in self._states:
                continue
            state = self._states[pid]
            for (ag_day, agreed_pid, tok_text) in agree_list:
                if ag_day != day:
                    continue
                # Find what the agreed-with player (agreed_pid) was ATTACKing
                agreed_target = self._last_attack_target.get(agreed_pid)
                state.d2_applicable += 1
                if agreed_target is not None and agreed_target in majority_set:
                    state.d2_followed += 1
                    state.d2_evidence.append(
                        (day, f"AGREE P{agreed_pid} -> target P{agreed_target} was majority")
                    )
                else:
                    state.d2_evidence.append(
                        (day, f"AGREE P{agreed_pid} -> target P{agreed_target} not majority")
                    )
        self._day_agree_tokens.clear()

        # D3: pressure (INQUIRE+ATTACK on p) and aggression (ATTACK by p)
        total_voters = max(len(vote_map), 1)
        for pid in self._states:
            if pid == self._self_id:
                continue
            pressure   = self._day_pressure.get(pid, {}).get(day, 0) / total_voters
            aggression = self._states[pid].d3_round_attacks.get(day, 0) / total_voters
            self._states[pid].d3_suspicion_series.append(pressure)
            self._states[pid].d3_accused_series.append(aggression)
            if pressure > 0 or aggression > 0:
                self._states[pid].d3_evidence.append(
                    (day, f"pressure={pressure:.2f} aggression={aggression:.2f}")
                )

    def observe_execution(self, day: int, agent_id: int, revealed_role: str) -> None:
        """
        Called when a player is eliminated and their role revealed.
        Updates D5 for DIVINED claims and COMINGOUT claims contradicted by the reveal.
        """
        actual_is_wolf = (revealed_role.upper() == "WEREWOLF")

        # D5: check DIVINED claims about this player
        for claimer, claims in self._seer_claims.items():
            for target, claimed_result in claims:
                if target != agent_id:
                    continue
                state = self._states.get(claimer)
                if state is None:
                    continue
                state.d5_total_claims += 1
                claimed_is_wolf = (claimed_result == "WEREWOLF")
                if actual_is_wolf != claimed_is_wolf:
                    state.d5_contradicted_claims += 1
                    state.d5_evidence.append((
                        day,
                        f"Claimed DIVINED P{agent_id} {claimed_result}, "
                        f"actual role {revealed_role}"
                    ))

        # D5: if eliminated player claimed a role via COMINGOUT that contradicts reveal
        claimed_role = self._comingout_claims.get(agent_id)
        if claimed_role is not None:
            state = self._states.get(agent_id)
            if state is not None:
                actual_is_seer = (revealed_role.upper() == "SEER")
                claimed_is_seer = (claimed_role == "SEER")
                if actual_is_seer != claimed_is_seer:
                    state.d5_contradicted_claims += 1
                    state.d5_evidence.append((
                        day,
                        f"Claimed COMINGOUT SEER, actual role {revealed_role}"
                    ))

    def observe_actual_divine(self, target: int, result: str) -> None:
        """
        Record the REAL divine result (only known if this agent is the Seer
        or can be inferred from an execution reveal).
        Used to validate D5 seer claims against ground truth.
        """
        self._true_divines[target] = result.upper()

    # ------------------------------------------------------------------ #
    #  Sigma update                                                        #
    # ------------------------------------------------------------------ #

    def end_of_day_update(
        self,
        day:           int,
        rnd:           int,
        true_roles:    dict,        # {pid: Role} — full ground truth for logging
        base_beliefs:  dict,        # {pid: float} — base agent's wolf probabilities
        decision_weights: dict,     # {"w1": float, "w2": float}
        alive_pids:    list,
    ) -> None:
        """
        Recompute sigma for all alive players using the asymmetric update rule.
        Appends one row per alive player to self._evolution_rows for CSV export.
        """
        from werewolf.const import Role as WRole

        for pid in alive_pids:
            if pid == self._self_id:
                continue
            state = self._states.get(pid)
            if state is None:
                continue

            d1 = compute_d1(state)
            d2 = compute_d2(state)
            d3 = compute_d3(state)
            d4 = compute_d4(state, self._single_wolf)
            d5 = compute_d5(state)
            E  = compute_E((d1, d2, d3, d4, d5), self._weights)

            old_sigma = self._sigma.get(pid, 0.5)
            # Threshold 0.15: D1=0.5 (E=0.15), D3=1.0 (E=0.20), D4=1.0 (E=0.20)
            # all push sigma upward; only very weak signals decrease it.
            if E > 0.15:
                new_sigma = min(1.0, old_sigma + 0.15 * E)
            else:
                new_sigma = max(0.0, old_sigma - 0.05 * (1.0 - E))
            delta = new_sigma - old_sigma
            self._sigma[pid] = new_sigma

            # Combined score
            w1 = decision_weights.get("w1", 0.6)
            w2 = decision_weights.get("w2", 0.4)
            belief = base_beliefs.get(pid, 0.5)
            combined = w1 * belief + w2 * new_sigma

            # True role for logging
            role_obj = true_roles.get(pid)
            role_str = role_obj.value if role_obj else "Unknown"

            self._evolution_rows.append({
                "day":           day,
                "round":         rnd,
                "target_player": f"P{pid}",
                "target_role":   role_str,
                "sigma":         round(new_sigma, 4),
                "trust":         round(1.0 - new_sigma, 4),
                "belief_wolf":   round(belief, 4),
                "combined_score": round(combined, 4),
                "d1":  round(d1, 4),
                "d2":  round(d2, 4),
                "d3":  round(d3, 4),
                "d4":  round(d4, 4),
                "d5":  round(d5, 4),
                "E":   round(E, 4),
                "delta_sigma": round(delta, 4),
            })

    # ------------------------------------------------------------------ #
    #  Query interface                                                     #
    # ------------------------------------------------------------------ #

    def get_sigma(self, player_id: int) -> float:
        return self._sigma.get(player_id, 0.5)

    def get_detectors(self, player_id: int) -> dict:
        """Return {d1, d2, d3, d4, d5, E} for the given player."""
        state = self._states.get(player_id)
        if state is None:
            return {"d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": 0.0, "d5": 0.0, "E": 0.0}
        d1 = compute_d1(state)
        d2 = compute_d2(state)
        d3 = compute_d3(state)
        d4 = compute_d4(state, self._single_wolf)
        d5 = compute_d5(state)
        E  = compute_E((d1, d2, d3, d4, d5), self._weights)
        return {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5, "E": E}

    def snapshot(
        self,
        day:          int,
        rnd:          int,
        base_beliefs: dict,
        decision_weights: dict,
        alive_pids:   list,
        decision_pid: int,
    ) -> dict:
        """
        Return a full reasoning snapshot for the suspicion trace log (spec §3.5).
        """
        w1 = decision_weights.get("w1", 0.6)
        w2 = decision_weights.get("w2", 0.4)

        if _DIAG:
            # Show raw state for first two alive others so we can see why D-values are 0
            sample_pids = [p for p in alive_pids if p != self._self_id][:2]
            print(f"\n[DIAG snapshot] day={day} rnd={rnd}  "
                  f"observe_votes_calls_so_far: check d1_rounds below")
            for sp in sample_pids:
                st = self._states.get(sp)
                if st:
                    print(f"  P{sp} raw state: "
                          f"d1_rounds={st.d1_rounds} d1_mismatches={st.d1_mismatches} "
                          f"intended_attack={st.intended_attack}  "
                          f"d2_applicable={st.d2_applicable} d2_followed={st.d2_followed}  "
                          f"d3_series_len={len(st.d3_suspicion_series)}  "
                          f"d4_rounds={st.d4_rounds} d4_defends={st.d4_defends}  "
                          f"d5_total={st.d5_total_claims}")

        detectors_per_player = {}
        combined_scores      = {}
        for pid in alive_pids:
            if pid == self._self_id:
                continue
            dvals = self.get_detectors(pid)
            detectors_per_player[pid] = dvals
            belief = base_beliefs.get(pid, 0.5)
            sigma  = self._sigma.get(pid, 0.5)
            combined_scores[pid] = round(w1 * belief + w2 * sigma, 4)

        # Collect per-player evidence lists for the tracer (Change 6)
        evidence_per_player = {}
        for pid in alive_pids:
            if pid == self._self_id:
                continue
            st = self._states.get(pid)
            if st:
                evidence_per_player[pid] = {
                    "d1": list(st.d1_evidence),
                    "d2": list(st.d2_evidence),
                    "d3": list(st.d3_evidence),
                    "d4": list(st.d4_evidence),
                    "d5": list(st.d5_evidence),
                }

        return {
            "day":              day,
            "round":            rnd,
            "bayesian_beliefs": {p: round(base_beliefs.get(p, 0.5), 4)
                                 for p in alive_pids if p != self._self_id},
            "sigma":            {p: round(self._sigma.get(p, 0.5), 4)
                                 for p in alive_pids if p != self._self_id},
            "detectors":        detectors_per_player,
            "combined_scores":  combined_scores,
            "evidence":         evidence_per_player,
            "decision":         decision_pid,
        }

    def get_evolution_rows(self) -> list:
        """Return the accumulated evolution rows for CSV export."""
        return list(self._evolution_rows)

    def update_weights(self, weights: dict) -> None:
        """Update detector weights at runtime (used by grid search)."""
        self._weights.update(weights)
