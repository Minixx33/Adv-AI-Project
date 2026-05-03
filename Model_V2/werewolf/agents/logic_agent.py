"""
werewolf/agents/logic_agent.py

LogicAgent — forward-chaining symbolic-reasoning agent.

Hierarchy:
  AbstractAgent
    └── LogicAgent
          ├── _KnowledgeBase   — stores literals and tagged hypotheticals
          └── _ForwardChainer  — inference engine (BFS rule application)

Design (spec §1 — Logic-Based Agent):
  • Maintains a KB of observed FACTS and CLAIMS.
  • Facts:    ground truths that follow logically from game events
              (e.g., a player was executed and revealed as Villager →
               Fact(IsVillager(P3))).
  • Claims:   utterances from other agents, stored as
              Claim(claimer=P2, predicate="Wolf(P5)") — NOT treated as facts.
  • Rules (forward chaining):
      R1: Fact(IsWolf(p)) → Vote(p)
      R2: Fact(IsVillager(p)) → Retract(Suspect(p))
      R3: claim SEER from p1 AND claim SEER from p2 (p1≠p2)
             → Fact(AtLeastOneWolfTeam({p1, p2}))
      R4: Claim(p, DIVINED(q, Wolf)) AND Fact(IsVillager(q))
             → Contradiction(p) → Fact(Suspicious(p))
      R5: Claim(p, DIVINED(q, Human)) AND Fact(IsWolf(q))
             → Contradiction(p) → Fact(Suspicious(p))
      R6: Fact(Suspicious(p)) → increase vote weight for p
  • Contradiction handling: if two players claim the same exclusive role,
    infer that at least one is on the wolf team (→ both become mild suspects).

Actions:
  vote()   → player with most accumulated vote-weight
  talk()   → "COMINGOUT P{self} {own_role}" on first day, then conclusions
  divine() → the player with the highest suspicion score not yet divined
  attack() → random from alive non-wolf (logic agent may be a wolf by role assignment)
  guard()  → player with no claims against them (most likely innocent)

NOT compatible with the suspicion module (spec §1):
  "deals in certainties, not degrees."  get_wolf_belief() raises.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Set

from ..agent import AbstractAgent
from ..gameinfo import GameInfo, GameSetting
from ..const import Role


# ------------------------------------------------------------------ #
#  KB data classes                                                    #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Fact:
    predicate: str   # e.g. "IsWolf", "IsVillager", "Suspicious", "AtLeastOneWolfTeam"
    subject:   object  # player_id (int) or frozenset of player_ids


@dataclass(frozen=True)
class Claim:
    claimer: int    # player_id making the claim
    predicate: str  # e.g. "Wolf", "Human", "Seer"
    subject: object # player_id or role string


# Talk parsers
_COMINGOUT_RE = re.compile(r"\bCOMINGOUT\s+P(\d+)\s+(\w+)", re.IGNORECASE)
_DIVINED_RE   = re.compile(r"\bDIVINED\s+P(\d+)\s+(WEREWOLF|HUMAN)", re.IGNORECASE)
_VOTE_RE      = re.compile(r"\bVOTE\s+P(\d+)", re.IGNORECASE)


class _KnowledgeBase:
    """Stores facts and claims, deduplicating on insertion."""

    def __init__(self):
        self.facts:  Set[Fact]  = set()
        self.claims: Set[Claim] = set()

    def add_fact(self, fact: Fact) -> bool:
        if fact not in self.facts:
            self.facts.add(fact)
            return True
        return False

    def add_claim(self, claim: Claim) -> bool:
        if claim not in self.claims:
            self.claims.add(claim)
            return True
        return False

    def has_fact(self, predicate: str, subject) -> bool:
        return Fact(predicate, subject) in self.facts

    def facts_with(self, predicate: str) -> list:
        return [f for f in self.facts if f.predicate == predicate]

    def claims_by(self, claimer: int) -> list:
        return [c for c in self.claims if c.claimer == claimer]

    def claims_with_predicate(self, predicate: str) -> list:
        return [c for c in self.claims if c.predicate == predicate]


class _ForwardChainer:
    """
    Applies inference rules until fixpoint.
    Returns a set of newly derived facts added in this round.
    """

    def __init__(self, kb: _KnowledgeBase):
        self._kb = kb

    def run(self) -> Set[Fact]:
        new_facts: Set[Fact] = set()
        changed = True
        while changed:
            changed = False
            for rule in [
                self._rule_dual_seer_claim,
                self._rule_divine_contradiction_wolf,
                self._rule_divine_contradiction_human,
            ]:
                added = rule()
                new_facts |= added
                if added:
                    changed = True
        return new_facts

    def _rule_dual_seer_claim(self) -> Set[Fact]:
        """R3: Two distinct players both claim Seer → at least one is wolf-team."""
        seer_claimers = [
            c.claimer for c in self._kb.claims_with_predicate("Seer")
        ]
        added: Set[Fact] = set()
        if len(seer_claimers) >= 2:
            pair = frozenset(seer_claimers[:2])
            f = Fact("AtLeastOneWolfTeam", pair)
            if self._kb.add_fact(f):
                added.add(f)
                # Mark both as mildly suspicious
                for pid in pair:
                    sf = Fact("Suspicious", pid)
                    if self._kb.add_fact(sf):
                        added.add(sf)
        return added

    def _rule_divine_contradiction_wolf(self) -> Set[Fact]:
        """R4: Claimed DIVINED(q, Wolf) but q is known Villager → Suspicious(claimer)."""
        added: Set[Fact] = set()
        for claim in self._kb.claims_with_predicate("DivinedWolf"):
            q = claim.subject
            if self._kb.has_fact("IsVillager", q):
                f = Fact("Suspicious", claim.claimer)
                if self._kb.add_fact(f):
                    added.add(f)
        return added

    def _rule_divine_contradiction_human(self) -> Set[Fact]:
        """R5: Claimed DIVINED(q, Human) but q is known Wolf → Suspicious(claimer)."""
        added: Set[Fact] = set()
        for claim in self._kb.claims_with_predicate("DivinedHuman"):
            q = claim.subject
            if self._kb.has_fact("IsWolf", q):
                f = Fact("Suspicious", claim.claimer)
                if self._kb.add_fact(f):
                    added.add(f)
        return added


class LogicAgent(AbstractAgent):
    """
    Symbolic reasoning agent with forward-chaining inference.
    NOT compatible with the suspicion module.
    """

    def __init__(self, agent_id: int, seed: int = None):
        super().__init__(agent_id, name="LogicAgent", seed=seed)
        self._kb        = _KnowledgeBase()
        self._chainer   = _ForwardChainer(self._kb)
        self._vote_weight: dict = defaultdict(float)  # {pid: float}
        self._seen_talks: set  = set()
        self._divined:    set  = set()
        self._announced_role   = False

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().initialize(game_info, game_setting)
        self._kb      = _KnowledgeBase()
        self._chainer = _ForwardChainer(self._kb)
        self._vote_weight  = defaultdict(float)
        self._seen_talks   = set()
        self._divined      = set()
        self._announced_role = False
        # Seed facts: own role is known
        own_role = game_info.role_map.get(self.agent_id)
        if own_role:
            self._kb.add_fact(Fact(f"Is{own_role.value.capitalize()}", self.agent_id))
        # Wolves know their teammates
        for pid, role in game_info.role_map.items():
            if role == Role.WEREWOLF:
                self._kb.add_fact(Fact("IsWolf", pid))

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().update(game_info, game_setting)
        self._ingest_talks(game_info.talk_list)
        self._ingest_reveals(game_info)
        self._ingest_night(game_info)
        self._chainer.run()
        self._recompute_vote_weights()

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_round_tokens(self) -> list:
        tokens = []
        own_role = self.my_role()

        # Role announcement (once per game)
        if not self._announced_role:
            self._announced_role = True
            if own_role and own_role not in (Role.WEREWOLF,):
                tokens.append(f"COMINGOUT P{self.agent_id} {own_role.value}")
                if own_role == Role.SEER and self.game_info and self.game_info.divine_result:
                    dr = self.game_info.divine_result
                    tokens.append(f"DIVINED P{dr.target} {dr.result}")

        # INQUIRE any known-suspicious alive player (one per round)
        for pid in self.alive_others():
            if self._kb.has_fact("Suspicious", pid):
                tokens.append(f"INQUIRE P{pid}")
                break

        # NOT for any alive player whose wolf claim contradicts known facts
        for claim in self._kb.claims_with_predicate("DivinedWolf"):
            t = claim.subject
            if isinstance(t, int) and t in self.alive_others() \
                    and self._kb.has_fact("IsVillager", t):
                tokens.append(f"NOT P{t} WEREWOLF")
                break

        # Vote + justification
        target = self._top_vote_target()
        if target is not None:
            tokens.append(f"VOTE P{target}")
            if self._kb.has_fact("Suspicious", target):
                tokens.append("BECAUSE claim_contradiction")
            elif self._kb.has_fact("IsWolf", target):
                tokens.append("BECAUSE seer_claim_conflict")

        tokens.append("Over")
        return tokens[:5]  # never exceed budget

    def vote(self) -> int:
        target = self._top_vote_target()
        return target if target is not None else self._rng.choice(
            self.alive_others() or [self.agent_id]
        )

    def attack(self) -> int:
        return self._random_non_wolf_target()

    def divine(self) -> int:
        """Divine the most suspicious undivined alive player."""
        candidates = [p for p in self.alive_others() if p not in self._divined]
        if not candidates:
            candidates = self.alive_others()
        if not candidates:
            return self.agent_id
        target = max(candidates, key=lambda p: self._vote_weight[p])
        self._divined.add(target)
        return target

    def guard(self) -> int:
        """Protect the player with zero or lowest suspicion."""
        others = self.alive_others()
        if not others:
            return self.agent_id
        return min(others, key=lambda p: self._vote_weight[p])

    def get_wolf_belief(self, player_id: int) -> float:
        raise NotImplementedError(
            "LogicAgent is not compatible with the suspicion module "
            "(it deals in certainties, not probabilities)."
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ingest_talks(self, talk_list: list) -> None:
        for talk in talk_list:
            key = (talk.day, talk.turn, talk.agent)
            if key in self._seen_talks:
                continue
            self._seen_talks.add(key)
            pid = talk.agent

            for m in _COMINGOUT_RE.finditer(talk.text):
                subject_pid = int(m.group(1))
                role_str    = m.group(2).upper()
                if subject_pid == pid:  # claiming own role
                    self._kb.add_claim(Claim(pid, role_str.capitalize(), pid))

            for m in _DIVINED_RE.finditer(talk.text):
                target  = int(m.group(1))
                result  = m.group(2).upper()
                pred    = "DivinedWolf" if result == "WEREWOLF" else "DivinedHuman"
                self._kb.add_claim(Claim(pid, pred, target))

    def _ingest_reveals(self, game_info: GameInfo) -> None:
        """Add facts for eliminated players whose roles are now public."""
        for pid, role in game_info.role_map.items():
            if game_info.status_map.get(pid) and game_info.status_map[pid].value == "DEAD":
                if role == Role.WEREWOLF:
                    self._kb.add_fact(Fact("IsWolf", pid))
                else:
                    self._kb.add_fact(Fact("IsVillager", pid))  # broad non-wolf category

    def _ingest_night(self, game_info: GameInfo) -> None:
        """If we are the Seer, add facts from our divine result."""
        from ..const import Species
        dr = game_info.divine_result
        if dr is None:
            return
        target = dr.target
        if dr.result == Species.WEREWOLF.value:
            self._kb.add_fact(Fact("IsWolf", target))
        else:
            self._kb.add_fact(Fact("IsVillager", target))

    def _recompute_vote_weights(self) -> None:
        """Map KB facts to vote weights for each player."""
        self._vote_weight = defaultdict(float)
        for f in self._kb.facts_with("IsWolf"):
            self._vote_weight[f.subject] += 2.0
        for f in self._kb.facts_with("Suspicious"):
            if isinstance(f.subject, int):
                self._vote_weight[f.subject] += 1.0
        for f in self._kb.facts_with("AtLeastOneWolfTeam"):
            if isinstance(f.subject, frozenset):
                for pid in f.subject:
                    self._vote_weight[pid] += 0.5

    def _top_vote_target(self) -> Optional[int]:
        alive = self.alive_others()
        if not alive:
            return None
        best = max(alive, key=lambda p: self._vote_weight[p])
        if self._vote_weight[best] == 0.0:
            return self._rng.choice(alive)
        return best
