"""
tests/test_suspicion.py

Suspicion module and EnhancedAgent tests (spec §8 requirements).

Tests:
  test_sigma_initializes_to_half            — σ(p) = 0.5 for all players
  test_d1_mismatch_detection                — vote-accusation mismatch counted
  test_d2_bandwagon_detection               — majority follower detected
  test_d5_claim_contradiction               — false seer claim increments D5
  test_asymmetric_update_high_evidence      — high E → σ increases
  test_asymmetric_update_low_evidence       — low E → σ decreases slowly
  test_enhanced_agent_never_votes_self      — EnhancedAgent.vote() ≠ self
  test_enhanced_agent_combined_score        — combined score = w1*belief + w2*σ
  test_enhanced_agent_with_bayesian_base    — full game with EnhancedAgent runs
  test_suspicion_not_wolf_constraint        — already in test_game; sanity repeat
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.const        import Role, Status, Team, ROLE_CONFIG
from werewolf.gameinfo     import GameInfo, GameSetting, Talk, Vote, Judge
from werewolf.game         import WerewolfGame
from werewolf.role_assigner import assign_roles
from werewolf.agents.bayesian_agent  import BayesianAgent
from werewolf.agents.heuristic_agent import HeuristicAgent
from suspicion.detectors   import (
    DetectorState, compute_d1, compute_d2, compute_d3, compute_d4,
    compute_d5, compute_E,
)
from suspicion.module      import SuspicionModule
from suspicion.enhanced_agent import EnhancedAgent


def _make_gameinfo(agent_id: int, n=8, role=Role.VILLAGER, talks=None) -> GameInfo:
    status = {i: Status.ALIVE for i in range(1, n + 1)}
    return GameInfo(day=1, agent=agent_id,
                    role_map={agent_id: role},
                    status_map=status, talk_list=talks or [])


def _make_setting(n=8) -> GameSetting:
    return GameSetting(player_num=n, role_num_map=dict(ROLE_CONFIG[n]))


class TestDetectors(unittest.TestCase):

    def test_d1_zero_initially(self):
        state = DetectorState()
        self.assertEqual(compute_d1(state), 0.0)

    def test_d1_mismatch_counted(self):
        state = DetectorState()
        state.d1_rounds    = 3
        state.d1_mismatches = 2
        self.assertAlmostEqual(compute_d1(state), 2/3)

    def test_d2_bandwagon(self):
        state = DetectorState()
        state.d2_applicable = 4
        state.d2_followed   = 3
        self.assertAlmostEqual(compute_d2(state), 0.75)

    def test_d5_false_claim(self):
        state = DetectorState()
        state.d5_total_claims        = 3
        state.d5_contradicted_claims = 2
        self.assertAlmostEqual(compute_d5(state), 2/3)

    def test_d3_positive_correlation(self):
        state = DetectorState()
        # pressure on p goes up, accusations by p go up → positive correlation
        state.d3_suspicion_series = [0.1, 0.3, 0.5, 0.7, 0.9]
        state.d3_accused_series   = [0.0, 0.3, 0.5, 0.8, 1.0]
        d3 = compute_d3(state)
        self.assertGreater(d3, 0.5)

    def test_d4_disabled_single_wolf(self):
        state = DetectorState()
        state.d4_defends = {2: 5}
        state.d4_rounds  = 5
        self.assertEqual(compute_d4(state, single_wolf_game=True), 0.0)

    def test_composite_E_weighted(self):
        weights = {"a1": 1.0, "a2": 0.0, "a3": 0.0, "a4": 0.0, "a5": 0.0}
        E = compute_E((0.8, 0.0, 0.0, 0.0, 0.0), weights)
        self.assertAlmostEqual(E, 0.8)


class TestSuspicionModule(unittest.TestCase):

    def test_sigma_initializes_half(self):
        module = SuspicionModule(player_ids=list(range(1, 9)), self_id=1)
        for pid in range(2, 9):
            self.assertAlmostEqual(module.get_sigma(pid), 0.5)

    def test_asymmetric_update_high_E(self):
        """High evidence → σ increases."""
        module = SuspicionModule(player_ids=[1, 2, 3], self_id=1)
        # Force high D1 for P2
        module._states[2].d1_rounds    = 4
        module._states[2].d1_mismatches = 4  # D1 = 1.0 → E ≈ a1 = 0.3 > threshold if other detectors also fire

        from werewolf.const import Role as WRole
        true_roles = {2: WRole.WEREWOLF, 3: WRole.VILLAGER}
        base_beliefs = {2: 0.8, 3: 0.2}
        module._states[2].d1_mismatches = 4  # D1 = 1.0
        # Set E > 0.5 by having all detectors fire
        module._states[2].d2_applicable = 4
        module._states[2].d2_followed   = 4  # D2 = 1.0
        module._states[2].d5_total_claims = 2
        module._states[2].d5_contradicted_claims = 2  # D5 = 1.0

        old_sigma = module.get_sigma(2)
        module.end_of_day_update(
            day=1, rnd=5, true_roles=true_roles,
            base_beliefs=base_beliefs,
            decision_weights={"w1": 0.6, "w2": 0.4},
            alive_pids=[2, 3],
        )
        self.assertGreater(module.get_sigma(2), old_sigma,
                           msg="High evidence should increase sigma")

    def test_asymmetric_update_low_E(self):
        """Low evidence → σ decreases (slow)."""
        module = SuspicionModule(player_ids=[1, 2, 3], self_id=1)
        from werewolf.const import Role as WRole
        true_roles = {2: WRole.VILLAGER, 3: WRole.WEREWOLF}
        base_beliefs = {2: 0.1, 3: 0.9}

        old_sigma = module.get_sigma(2)  # 0.5
        module.end_of_day_update(
            day=1, rnd=5, true_roles=true_roles,
            base_beliefs=base_beliefs,
            decision_weights={"w1": 0.6, "w2": 0.4},
            alive_pids=[2, 3],
        )
        # E = 0 (no detectors fired) → should decrease
        self.assertLessEqual(module.get_sigma(2), old_sigma,
                              msg="Zero evidence should decrease sigma")

    def test_d5_updated_by_execution(self):
        module = SuspicionModule(player_ids=list(range(1, 9)), self_id=1)
        # P2 claimed Seer and divined P3 as wolf
        module._seer_claims[2].append((3, "WEREWOLF"))
        # P3 is executed and revealed as Villager → contradiction
        module.observe_execution(day=1, agent_id=3, revealed_role="VILLAGER")
        self.assertEqual(module._states[2].d5_total_claims, 1)
        self.assertEqual(module._states[2].d5_contradicted_claims, 1)


class TestEnhancedAgent(unittest.TestCase):

    def _make_enhanced(self, n=8):
        base = BayesianAgent(1, seed=0)
        enh  = EnhancedAgent(base)
        gi   = _make_gameinfo(1, n=n)
        enh.initialize(gi, _make_setting(n))
        return enh

    def test_vote_never_self(self):
        enh = self._make_enhanced()
        for _ in range(20):
            v = enh.vote()
            self.assertNotEqual(v, 1)

    def test_vote_returns_alive_player(self):
        enh = self._make_enhanced()
        v   = enh.vote()
        self.assertIn(v, range(1, 9))

    def test_combined_score_formula(self):
        """score = w1*belief + w2*sigma; vote should pick argmax."""
        enh = self._make_enhanced()
        # Manually inflate sigma for player 5
        enh._suspicion._sigma[5] = 0.95
        enh._base._beliefs[5]    = 0.9
        v = enh.vote()
        self.assertEqual(v, 5, msg="Should vote P5 with highest combined score")

    def test_full_game_with_enhanced(self):
        """Full game with one enhanced Bayesian agent runs to completion."""
        for seed in range(5):
            rng        = random.Random(seed)
            player_ids = list(range(1, 9))
            base = BayesianAgent(1, seed=seed)
            agents = {
                1: EnhancedAgent(base),
                **{i: HeuristicAgent(i, seed=seed + i) for i in range(2, 9)},
            }
            roles = assign_roles(
                player_ids, 8, rng, suspicion_player_id=1
            )
            game = WerewolfGame(agents, roles, game_id=seed, seed=seed,
                                suspicion_agent_ids=[1])
            rec  = game.run()
            self.assertIn(rec["winner"], (Team.VILLAGE, Team.WOLF),
                          msg=f"Enhanced agent game failed at seed={seed}")


class TestDetectorPipeline(unittest.TestCase):
    """
    Integration tests: feed crafted Talk/Vote data through the full pipeline
    (observe_talks → observe_votes → observe_execution) and assert each
    detector's computed value.  No state is set directly.
    """

    def _module(self, player_ids=None, self_id=8, n_wolves=2):
        if player_ids is None:
            player_ids = list(range(1, 9))
        return SuspicionModule(player_ids=player_ids, self_id=self_id,
                               n_wolves=n_wolves)

    def _talk(self, agent, text, day=1, turn=0):
        return Talk(day=day, turn=turn, agent=agent, text=text)

    # ── D1: Vote-Accusation Mismatch ──────────────────────────────────────

    def test_d1_fires_on_attack_vote_mismatch(self):
        """P2 ATTACKs P3 but actually votes P4 → D1 = 1.0."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "ATTACK P3")])
        mod.observe_votes(1, {2: 4, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertAlmostEqual(mod.get_detectors(2)["d1"], 1.0)

    def test_d1_zero_when_attack_matches_vote(self):
        """P2 ATTACKs P3 and votes P3 → D1 = 0.0 (no mismatch)."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "ATTACK P3")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertAlmostEqual(mod.get_detectors(2)["d1"], 0.0)

    def test_d1_zero_when_no_attack_token(self):
        """P2 votes without emitting ATTACK → D1 = 0.0 (no rounds counted)."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "VOTE P3")])
        mod.observe_votes(1, {2: 5, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertAlmostEqual(mod.get_detectors(2)["d1"], 0.0)

    # ── D2: Bandwagon Index ───────────────────────────────────────────────

    def test_d2_fires_when_agree_joins_majority(self):
        """P2 AGREEs with P3 who was ATTACKing P4; P4 gets majority → D2 = 1.0."""
        mod = self._module()
        mod.observe_talks(1, [
            self._talk(3, "ATTACK P4"),
            self._talk(2, "AGREE P3"),
        ])
        # P4 receives 3 votes → majority
        mod.observe_votes(1, {2: 4, 3: 4, 5: 4, 6: 5, 7: 5})
        self.assertAlmostEqual(mod.get_detectors(2)["d2"], 1.0)

    def test_d2_zero_when_agree_not_majority(self):
        """P2 AGREEs with P3 who ATTACKed P4; P4 does NOT get majority → D2 = 0.0."""
        mod = self._module()
        mod.observe_talks(1, [
            self._talk(3, "ATTACK P4"),
            self._talk(2, "AGREE P3"),
        ])
        # P5 gets majority, not P4
        mod.observe_votes(1, {2: 5, 3: 5, 4: 5, 5: 4, 6: 4, 7: 3})
        self.assertAlmostEqual(mod.get_detectors(2)["d2"], 0.0)

    def test_d2_zero_when_no_agree_token(self):
        """P2 never emits AGREE → D2 = 0.0."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "ATTACK P3")])
        mod.observe_votes(1, {2: 3, 3: 4, 4: 4, 5: 4, 6: 5, 7: 5})
        self.assertAlmostEqual(mod.get_detectors(2)["d2"], 0.0)

    # ── D3: Pressure-Triggered Accusations ───────────────────────────────

    def test_d3_n1_aggression_fallback(self):
        """Day 1 only (n=1 series): D3 = min(1, aggression/voters) > 0 when P2 attacks."""
        mod = self._module()
        mod.observe_talks(1, [
            self._talk(2, "ATTACK P3"),
            self._talk(2, "ATTACK P4"),
        ])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertGreater(mod.get_detectors(2)["d3"], 0.0)

    def test_d3_zero_when_no_aggression(self):
        """P2 never emits ATTACK; zero aggression → D3 = 0.0 on Day 1."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "VOTE P3")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertAlmostEqual(mod.get_detectors(2)["d3"], 0.0)

    def test_d3_pearson_n2_positive(self):
        """Two days: pressure and aggression both increase → Pearson D3 > 0."""
        mod = self._module()
        # Day 1: P2 emits 1 ATTACK; nobody attacks P2
        mod.observe_talks(1, [self._talk(2, "ATTACK P3")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        # Day 2: P2 emits 3 ATTACKs; P3 and P4 attack P2 (higher pressure)
        mod.observe_talks(2, [
            self._talk(2, "ATTACK P3"),
            self._talk(2, "ATTACK P4"),
            self._talk(2, "ATTACK P5"),
            self._talk(3, "ATTACK P2"),
            self._talk(4, "ATTACK P2"),
        ])
        mod.observe_votes(2, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertGreater(mod.get_detectors(2)["d3"], 0.0)

    # ── D4: Protection Patterns ───────────────────────────────────────────

    def test_d4_fires_on_defend_token(self):
        """P2 emits DEFEND P5 in a multi-wolf game → D4 > 0."""
        mod = self._module(n_wolves=2)
        mod.observe_talks(1, [self._talk(2, "DEFEND P5")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertGreater(mod.get_detectors(2)["d4"], 0.0)

    def test_d4_fires_on_not_werewolf_token(self):
        """P2 emits NOT P5 WEREWOLF (defending P5) → D4 > 0."""
        mod = self._module(n_wolves=2)
        mod.observe_talks(1, [self._talk(2, "NOT P5 WEREWOLF")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertGreater(mod.get_detectors(2)["d4"], 0.0)

    def test_d4_ignores_self_defense(self):
        """P2 emits NOT P2 WEREWOLF (own self-defense) → excluded → D4 = 0.0."""
        mod = self._module(n_wolves=2)
        mod.observe_talks(1, [self._talk(2, "NOT P2 WEREWOLF")])
        mod.observe_votes(1, {2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4})
        self.assertAlmostEqual(mod.get_detectors(2)["d4"], 0.0)

    def test_d4_disabled_in_single_wolf_game(self):
        """D4 returns 0 for single-wolf (5-player) games regardless of tokens."""
        mod = self._module(player_ids=list(range(1, 6)), self_id=5, n_wolves=1)
        mod.observe_talks(1, [self._talk(2, "DEFEND P3")])
        mod.observe_votes(1, {1: 2, 2: 3, 3: 2, 4: 2})
        self.assertAlmostEqual(mod.get_detectors(2)["d4"], 0.0)

    # ── D5: Claim Consistency ─────────────────────────────────────────────

    def test_d5_fires_on_divined_contradiction(self):
        """P2 claims DIVINED P3 WEREWOLF; P3 revealed as VILLAGER → D5 = 1.0."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "DIVINED P3 WEREWOLF")])
        mod.observe_execution(1, agent_id=3, revealed_role="VILLAGER")
        self.assertAlmostEqual(mod.get_detectors(2)["d5"], 1.0)

    def test_d5_zero_when_divined_claim_correct(self):
        """P2 claims DIVINED P3 WEREWOLF; P3 revealed as WEREWOLF → D5 = 0.0."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "DIVINED P3 WEREWOLF")])
        mod.observe_execution(1, agent_id=3, revealed_role="WEREWOLF")
        self.assertAlmostEqual(mod.get_detectors(2)["d5"], 0.0)

    def test_d5_comingout_increments_total_claims(self):
        """P2 emits COMINGOUT P2 SEER → d5_total_claims = 1 immediately."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "COMINGOUT P2 SEER")])
        self.assertEqual(mod._states[2].d5_total_claims, 1)

    def test_d5_comingout_contradiction_on_execution(self):
        """P2 claims SEER via COMINGOUT but is revealed WEREWOLF → contradiction logged."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "COMINGOUT P2 SEER")])
        mod.observe_execution(1, agent_id=2, revealed_role="WEREWOLF")
        state = mod._states[2]
        self.assertEqual(state.d5_total_claims, 1)
        self.assertEqual(state.d5_contradicted_claims, 1)

    def test_d5_comingout_no_contradiction_when_role_matches(self):
        """P2 claims COMINGOUT P2 SEER and IS revealed as SEER → no contradiction."""
        mod = self._module()
        mod.observe_talks(1, [self._talk(2, "COMINGOUT P2 SEER")])
        mod.observe_execution(1, agent_id=2, revealed_role="SEER")
        state = mod._states[2]
        self.assertEqual(state.d5_total_claims, 1)
        self.assertEqual(state.d5_contradicted_claims, 0)


if __name__ == "__main__":
    unittest.main()
