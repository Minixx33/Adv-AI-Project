"""werewolf.agents.suspicion_agent — Agent 4: two-layer suspicion model.

Architecture overview
---------------------
Layer 1 — Bayesian belief
    ``belief[p] = P(p is Werewolf)``; updated identically to BayesianAgent.

Layer 2 — Behavioural suspicion
    ``sigma[p] ∈ [0, 1]``; updated from five detectors that recognise
    deceptive patterns independent of explicit wolf accusations.

Combined decision score
    ``score[p] = 0.6 * belief[p] + 0.4 * sigma[p]``

All voting, divine, and attack decisions are based on ``score``, not on
``belief`` alone.

Detectors
---------
D1 — Vote-Accusation Mismatch
    Did the player say they would vote for X but actually vote for Y ≠ X?
    ``D1[p] = mismatch_count[p] / rounds_played[p]``

D2 — Bandwagon Index
    Does the player always follow the majority vote (eliminate the likely
    target regardless of evidence)?
    ``D2[p] = times_followed_majority[p] / applicable_rounds[p]``

D3 — Pressure-Triggered Accusations
    Does the player increase their accusations specifically when *they*
    are under high suspicion (diversion tactic)?
    ``D3[p] = pressure_accuse[p] / rounds_played[p]``
    where a round counts only if ``sigma[p] > 0.6`` at vote time.

D4 — Protection Patterns
    Does the player consistently defend a particular other player?
    ``D4[p] = max_q(defense_count[p][q]) / rounds_played[p]``

D5 — Claim Consistency
    Does the player ever contradict their own COMINGOUT role claim?
    ``D5[p] = contradictions[p] / total_claims[p]``

Composite evidence and sigma update
------------------------------------
``E[p] = 0.30*D1 + 0.15*D2 + 0.20*D3 + 0.20*D4 + 0.15*D5``

Sigma is updated asymmetrically after each voting round — trust is lost
faster than it is gained::

    if E[p] > 0.5:
        sigma[p] = min(1.0, sigma[p] + 0.15 * E[p])   # fast increase
    else:
        sigma[p] = max(0.0, sigma[p] - 0.05 * (1-E[p]))  # slow decrease

Module-level constants
----------------------
_VOTE_DELTA, _ESTIMATE_WOLF_DELTA, _DEFEND_DELTA,
_DIVINE_WOLF, _DIVINE_HUMAN  — same as BayesianAgent.
_W_BELIEF = 0.6, _W_SIGMA = 0.4  — score layer weights.
_W_D1..._W_D5 — detector weights for E[p].

Classes
-------
SuspicionAgent : The primary research contribution; two-layer model.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import Agent, GameInfo, GameSetting, Talk, parse_talk, AGENT_NONE
from werewolf.const import Role, Species, Status


# ── Bayesian parameters (identical to BayesianAgent) ────────────────────────
_VOTE_DELTA:          float = 0.05
_ESTIMATE_WOLF_DELTA: float = 0.10
_DEFEND_DELTA:        float = 0.04
_DIVINE_WOLF:         float = 0.90
_DIVINE_HUMAN:        float = 0.08

# ── Combined score weights ───────────────────────────────────────────────────
_W_BELIEF: float = 0.6
_W_SIGMA:  float = 0.4

# ── Evidence composite weights (must sum to 1.0) ─────────────────────────��───
_W_D1: float = 0.30
_W_D2: float = 0.15
_W_D3: float = 0.20
_W_D4: float = 0.20
_W_D5: float = 0.15


class SuspicionAgent(AbstractAgent):
    """Agent 4 — two-layer (Bayesian + behavioural suspicion) model.

    Strategy summary
    ----------------
    * **talk()** : announces ``"VOTE <argmax score>"`` — combines belief
      and behavioural suspicion to identify the top target.
    * **vote()** : votes for ``argmax(score)`` among alive opponents.
    * **divine()**: investigates ``argmax(score)`` — either confirms the
      Bayesian belief or catches a behaviourally suspicious player.
    * **attack()**: kills ``argmin(score)`` among alive non-wolf players —
      eliminates the village's most-trusted member.

    Attributes — Layer 1 (Bayesian)
    --------------------------------
    _belief        : dict[Agent, float]
        ``P(p is Werewolf)`` for each non-self player.

    Attributes — Layer 2 (Suspicion)
    ---------------------------------
    _sigma         : dict[Agent, float]
        Behavioural suspicion score ∈ [0, 1]; initialised to 0.5.

    Attributes — D1 (Vote-Accusation Mismatch)
    -------------------------------------------
    _declared_vote  : dict[Agent, Optional[Agent]]
        Last ``VOTE X`` talk each player made this round.
    _mismatch_count : dict[Agent, int]
        Times player's actual vote differed from declared vote.
    _d1_rounds      : dict[Agent, int]
        Total rounds used for D1 denominator.

    Attributes — D2 (Bandwagon Index)
    -----------------------------------
    _bandwagon_count : dict[Agent, int]
        Times player voted for the eventual executed agent.
    _d2_rounds       : dict[Agent, int]
        Total rounds used for D2 denominator.

    Attributes — D3 (Pressure-Triggered Accusations)
    --------------------------------------------------
    _pressure_accuse    : dict[Agent, int]
        Times player accused someone while their own sigma exceeded 0.6.
    _d3_rounds          : dict[Agent, int]
        Total rounds used for D3 denominator.
    _accused_this_round : dict[Agent, bool]
        Whether each player made any accusation in the current round.

    Attributes — D4 (Protection Patterns)
    ----------------------------------------
    _defense_count : dict[Agent, dict[Agent, int]]
        ``_defense_count[p][q]`` = times p defended q via ESTIMATE VILLAGER/SEER.
    _d4_rounds     : dict[Agent, int]
        Total rounds used for D4 denominator.

    Attributes — D5 (Claim Consistency)
    -------------------------------------
    _role_claims   : dict[Agent, str]
        Most recent COMINGOUT role claim per player.
    _contradictions: dict[Agent, int]
        Times a player changed their role claim.
    _claim_count   : dict[Agent, int]
        Total COMINGOUT events per player.

    Attributes — Misc
    ------------------
    _talk_head          : int               — Index of next unprocessed talk.
    _agent_lookup       : dict[str, Agent]  — String → Agent for parsing.
    _votes_cast         : int
    _correct_wolf_votes : int
    _false_accusations  : int
    """

    def __init__(self) -> None:
        self._game_info: GameInfo = None          # type: ignore[assignment]
        self._my_role: Role = None                # type: ignore[assignment]
        self._agent_lookup: dict[str, Agent]      = {}

        # Layer 1
        self._belief: dict[Agent, float]          = {}

        # Layer 2
        self._sigma: dict[Agent, float]           = {}

        # D1
        self._declared_vote: dict[Agent, Optional[Agent]] = {}
        self._mismatch_count: dict[Agent, int]    = defaultdict(int)
        self._d1_rounds: dict[Agent, int]         = defaultdict(int)

        # D2
        self._bandwagon_count: dict[Agent, int]   = defaultdict(int)
        self._d2_rounds: dict[Agent, int]         = defaultdict(int)

        # D3
        self._pressure_accuse: dict[Agent, int]   = defaultdict(int)
        self._d3_rounds: dict[Agent, int]         = defaultdict(int)
        self._accused_this_round: dict[Agent, bool] = defaultdict(bool)

        # D4
        self._defense_count: dict[Agent, dict[Agent, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._d4_rounds: dict[Agent, int]         = defaultdict(int)

        # D5
        self._role_claims: dict[Agent, str]       = {}
        self._contradictions: dict[Agent, int]    = defaultdict(int)
        self._claim_count: dict[Agent, int]       = defaultdict(int)

        self._talk_head: int                      = 0
        self._votes_cast: int                     = 0
        self._correct_wolf_votes: int             = 0
        self._false_accusations: int              = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Initialise both belief and sigma layers plus all detector accumulators.

        Args:
            game_info:    Day-0 snapshot; enumerates all players.
            game_setting: Static game configuration (unused by this agent).
        """
        self._game_info    = game_info
        self._my_role      = game_info.my_role
        self._agent_lookup = {str(a): a for a in game_info.status_map}
        others = [a for a in game_info.status_map if a != game_info.agent]
        prior  = 1.0 / max(len(others), 1)

        self._belief = {a: prior for a in others}
        self._sigma  = {a: 0.5   for a in others}

        for a in others:
            self._declared_vote[a]       = None
            self._accused_this_round[a]  = False

        self._talk_head          = 0
        self._votes_cast         = 0
        self._correct_wolf_votes = 0
        self._false_accusations  = 0

    def update(self, game_info: GameInfo) -> None:
        """Store snapshot and process new talks for both layers.

        Args:
            game_info: Current game state snapshot.
        """
        self._game_info = game_info
        self._process_new_talks()

    def day_start(self) -> None:
        """Apply divine results, zero dead players, and reset round accumulators.

        Called once per day before the first talk slot:

        * Divine result (Seer only): hard-sets belief *and* adjusts sigma.
        * Dead players: zeroed in ``_belief`` so they are excluded from argmax.
        * D1 / D3 per-round tracking: ``_declared_vote`` and
          ``_accused_this_round`` are reset to their initial state.
        """
        self._talk_head = 0
        gi = self._game_info

        if gi.divine_result:
            dr = gi.divine_result
            if dr.target in self._belief:
                if dr.result == Species.WEREWOLF:
                    self._belief[dr.target] = _DIVINE_WOLF
                    self._sigma[dr.target]  = self._clamp(
                        self._sigma[dr.target] + 0.3
                    )
                else:
                    self._belief[dr.target] = _DIVINE_HUMAN
                    self._sigma[dr.target]  = self._clamp(
                        self._sigma[dr.target] - 0.15
                    )

        for agent, status in gi.status_map.items():
            if status == Status.DEAD and agent in self._belief:
                self._belief[agent] = 0.0

        for a in self._declared_vote:
            self._declared_vote[a]      = None
            self._accused_this_round[a] = False

    # ------------------------------------------------------------------
    # Internal — talk processing
    # ------------------------------------------------------------------

    def _clamp(self, v: float) -> float:
        """Clamp *v* to [0.0, 1.0].

        Args:
            v: Raw floating-point value.

        Returns:
            ``v`` constrained to ``[0.0, 1.0]``.
        """
        return max(0.0, min(1.0, v))

    def _score(self, a: Agent) -> float:
        """Compute the combined decision score for agent *a*.

        Args:
            a: The agent being scored.

        Returns:
            ``_W_BELIEF * belief[a] + _W_SIGMA * sigma[a]``.
        """
        return _W_BELIEF * self._belief.get(a, 0.0) + _W_SIGMA * self._sigma.get(a, 0.5)

    def _process_new_talks(self) -> None:
        """Update both layers from talks that have not yet been processed.

        For each new talk this method:

        1. Updates ``_belief`` via Bayesian rules (same as BayesianAgent).
        2. Records D1 declared-vote tracking.
        3. Records D3 accusation-this-round flag.
        4. Records D4 defense counts.
        5. Records D5 COMINGOUT claim consistency.
        """
        gi = self._game_info
        for i in range(self._talk_head, len(gi.talk_list)):
            tk:     Talk             = gi.talk_list[i]
            talker: Agent            = tk.agent
            if talker == gi.agent or talker not in self._belief:
                continue

            parsed   = parse_talk(tk.text, self._agent_lookup)
            topic    = parsed.get("topic")
            target:  Optional[Agent] = parsed.get("target")
            role_str: str            = parsed.get("role", "")

            # ── Layer 1: Bayesian update ──────────────────────────────
            if target and target in self._belief:
                if topic == "VOTE":
                    self._belief[target] = self._clamp(
                        self._belief[target] + _VOTE_DELTA
                    )
                elif topic == "ESTIMATE":
                    if role_str == "WEREWOLF":
                        self._belief[target] = self._clamp(
                            self._belief[target] + _ESTIMATE_WOLF_DELTA
                        )
                    elif role_str in ("VILLAGER", "SEER"):
                        self._belief[target] = self._clamp(
                            self._belief[target] - _DEFEND_DELTA
                        )
                elif topic == "DIVINED":
                    result_str = parsed.get("result", "")
                    if result_str == "WEREWOLF":
                        self._belief[target] = _DIVINE_WOLF
                    elif result_str == "HUMAN":
                        self._belief[target] = _DIVINE_HUMAN

            # ── D1: record latest vote declaration ────────────────────
            if topic == "VOTE" and target:
                self._declared_vote[talker] = target

            # ── D3: flag any accusation made this round ───────────────
            if target and (
                topic == "VOTE"
                or (topic == "ESTIMATE" and role_str == "WEREWOLF")
            ):
                self._accused_this_round[talker] = True

            # ── D4: record defences ───────────────────────────────────
            if topic == "ESTIMATE" and target and target != talker:
                if role_str in ("VILLAGER", "SEER"):
                    self._defense_count[talker][target] += 1

            # ── D5: track COMINGOUT claim consistency ─────────────────
            if topic == "COMINGOUT" and target == talker:
                prev = self._role_claims.get(talker)
                if prev and prev != role_str:
                    self._contradictions[talker] += 1
                self._role_claims[talker] = role_str
                self._claim_count[talker] += 1

        self._talk_head = len(gi.talk_list)

    # ------------------------------------------------------------------
    # Post-vote sigma update (called by WerewolfGame)
    # ------------------------------------------------------------------

    def record_vote_round(
        self,
        actual_votes: dict[Agent, Agent],
        executed: Optional[Agent],
        true_roles: dict[Agent, Role],
    ) -> None:
        """Compute detector values and update sigma after each voting round.

        Called by ``WerewolfGame._notify_suspicion_vote_round()`` immediately
        after votes are resolved.  Only alive players are updated.

        The five detectors are evaluated and combined into composite evidence
        ``E[p]``, which drives the asymmetric sigma update::

            if E[p] > 0.5:  sigma += 0.15 * E[p]   (fast trust loss)
            else:           sigma -= 0.05 * (1-E[p]) (slow trust gain)

        Args:
            actual_votes: Mapping from each player to their actual vote target
                          this round.
            executed:     The player eliminated by the vote, or None.
            true_roles:   Ground-truth role map (used for vote-accuracy stats).
        """
        gi = self._game_info

        for p in list(self._sigma.keys()):
            if gi.status_map.get(p) != Status.ALIVE:
                continue

            declared = self._declared_vote.get(p)
            actual   = actual_votes.get(p)

            # D1 — Vote-Accusation Mismatch
            self._d1_rounds[p] += 1
            if declared is not None and actual is not None and declared != actual:
                self._mismatch_count[p] += 1
            d1 = self._mismatch_count[p] / self._d1_rounds[p]

            # D2 — Bandwagon Index
            self._d2_rounds[p] += 1
            if executed is not None and actual == executed:
                self._bandwagon_count[p] += 1
            d2 = self._bandwagon_count[p] / self._d2_rounds[p]

            # D3 — Pressure-Triggered Accusations
            self._d3_rounds[p] += 1
            if (
                self._sigma.get(p, 0.5) > 0.6
                and self._accused_this_round.get(p)
            ):
                self._pressure_accuse[p] += 1
            d3 = self._pressure_accuse[p] / self._d3_rounds[p]

            # D4 — Protection Patterns
            self._d4_rounds[p] += 1
            max_def = max(self._defense_count[p].values(), default=0)
            d4 = max_def / self._d4_rounds[p]

            # D5 — Claim Consistency
            total_claims = self._claim_count[p]
            d5 = self._contradictions[p] / total_claims if total_claims > 0 else 0.0

            # Composite evidence
            e = _W_D1 * d1 + _W_D2 * d2 + _W_D3 * d3 + _W_D4 * d4 + _W_D5 * d5

            # Asymmetric sigma update
            if e > 0.5:
                self._sigma[p] = self._clamp(self._sigma[p] + 0.15 * e)
            else:
                self._sigma[p] = self._clamp(self._sigma[p] - 0.05 * (1.0 - e))

        # Vote-accuracy stats
        if executed is not None and true_roles:
            role = true_roles.get(executed)
            if role == Role.WEREWOLF:
                self._correct_wolf_votes += 1
            elif role is not None:
                self._false_accusations += 1

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _argmax_score(self) -> Agent:
        """Return the alive opponent with the highest combined score.

        Returns:
            ``argmax(score)`` over alive non-self players.
        """
        gi         = self._game_info
        candidates = self._alive_others(gi)
        if not candidates:
            return gi.agent
        return max(candidates, key=self._score)

    def _argmin_score(self) -> Agent:
        """Return the alive non-wolf opponent with the lowest combined score.

        Used for the Werewolf attack decision: eliminate the player the
        village suspects least (most trusted = greatest threat).

        Returns:
            ``argmin(score)`` over alive non-wolf-team players.
        """
        gi        = self._game_info
        wolf_team = set(gi.role_map.keys())
        candidates = [a for a in gi.alive_agent_list if a not in wolf_team]
        if not candidates:
            candidates = self._alive_others(gi)
        if not candidates:
            return gi.agent
        return min(candidates, key=self._score)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def talk(self) -> str:
        """Announce a vote intention for the current highest-score player.

        Returns:
            ``"VOTE Agent[XX]"`` for ``argmax(score)`` among alive opponents.
        """
        return f"VOTE {self._argmax_score()}"

    def vote(self) -> Agent:
        """Vote for the alive player with the highest combined suspicion score.

        Returns:
            ``argmax(score)`` among alive opponents.
        """
        target = self._argmax_score()
        self._votes_cast += 1
        return target

    def whisper(self) -> str:
        """No-op wolf whisper.

        Returns:
            The string ``"SKIP"``.
        """
        return "SKIP"

    def attack(self) -> Agent:
        """Kill the alive non-wolf player with the lowest combined score.

        Returns:
            ``argmin(score)`` among alive non-wolf-team players.
        """
        return self._argmin_score()

    def divine(self) -> Agent:
        """Investigate the current highest-score player.

        Returns:
            ``argmax(score)`` among alive opponents.
        """
        return self._argmax_score()

    def finish(self) -> None:
        """No-op — this agent needs no end-of-game cleanup."""

    # ------------------------------------------------------------------
    # Stats hook
    # ------------------------------------------------------------------

    def record_vote_outcome(
        self, target: Agent, true_roles: dict[Agent, Role]
    ) -> None:
        """No-op — vote accuracy is already tracked in ``record_vote_round``."""

    def get_stats(self) -> dict:
        """Return performance metrics for CSV logging.

        ``final_belief_wolf`` is the maximum wolf belief at game end;
        ``final_sigma_wolf`` is the maximum behavioural suspicion score.
        Both serve as calibration proxies: a well-calibrated agent should
        have the true Werewolf at the top of both distributions.

        Returns:
            Dict with keys: ``votes_cast``, ``correct_wolf_votes``,
            ``false_accusations``, ``final_belief_wolf`` (float or None),
            ``final_sigma_wolf`` (float or None).
        """
        gi         = self._game_info
        candidates = self._alive_others(gi) if gi else []
        max_belief = max(
            (self._belief.get(a, 0.0) for a in candidates), default=None
        )
        max_sigma  = max(
            (self._sigma.get(a, 0.5) for a in candidates), default=None
        )
        return {
            "votes_cast":           self._votes_cast,
            "correct_wolf_votes":   self._correct_wolf_votes,
            "false_accusations":    self._false_accusations,
            "final_belief_wolf":    round(max_belief, 4) if max_belief is not None else None,
            "final_sigma_wolf":     round(max_sigma,  4) if max_sigma  is not None else None,
        }
