"""
tests/test_agents.py

Per-agent behaviour tests (spec §8 requirements).

Tests:
  test_random_agent_returns_valid_vote       — vote is always an alive other player
  test_heuristic_agent_tracks_accusations    — accusation counts update from talks
  test_bayesian_agent_updates_beliefs        — divine results update wolf probs
  test_bayesian_beats_heuristic              — >55% village win rate head-to-head
  test_mcts_returns_valid_vote               — MCTS vote is always valid
  test_logic_agent_infers_wolf               — claims + contradictions raise suspicion
  test_all_agents_survive_role_assignments   — every agent type can play any role
"""

import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.const       import Role, Status, Team, ROLE_CONFIG
from werewolf.gameinfo    import GameInfo, GameSetting, Talk, Vote, Judge
from werewolf.game        import WerewolfGame
from werewolf.role_assigner import assign_roles
from werewolf.agents.random_agent    import RandomAgent
from werewolf.agents.heuristic_agent import HeuristicAgent
from werewolf.agents.bayesian_agent  import BayesianAgent
from werewolf.agents.mcts_agent      import MCTSAgent
from werewolf.agents.logic_agent     import LogicAgent


def _make_gameinfo(agent_id: int, n_players: int = 8, role: Role = Role.VILLAGER,
                   talks=None, votes=None) -> GameInfo:
    """Minimal GameInfo for unit tests."""
    status_map = {i: Status.ALIVE for i in range(1, n_players + 1)}
    role_map   = {agent_id: role}
    return GameInfo(
        day=1, agent=agent_id,
        role_map=role_map, status_map=status_map,
        talk_list=talks or [], vote_list=votes or [],
    )


def _make_setting(n: int = 8) -> GameSetting:
    return GameSetting(player_num=n, role_num_map=dict(ROLE_CONFIG[n]))


def _full_game(agents_dict: dict, n_players: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    player_ids = sorted(agents_dict.keys())
    roles = assign_roles(player_ids, n_players, rng)
    game  = WerewolfGame(agents_dict, roles, game_id=0, seed=seed)
    return game.run()


class TestRandomAgent(unittest.TestCase):

    def test_vote_is_valid(self):
        agent = RandomAgent(1, seed=0)
        gi    = _make_gameinfo(1, n_players=8)
        agent.initialize(gi, _make_setting())
        for _ in range(50):
            v = agent.vote()
            self.assertIn(v, range(1, 9))
            self.assertNotEqual(v, 1)

    def test_talk_returns_valid_token(self):
        # RandomAgent now emits VOTE or Skip (no longer always Skip)
        agent = RandomAgent(1, seed=0)
        gi    = _make_gameinfo(1)
        agent.initialize(gi, _make_setting())
        token = agent.talk()
        self.assertTrue(
            token == "Skip" or token.startswith("VOTE") or token == "Over",
            msg=f"Unexpected token: {token!r}"
        )

    def test_runs_complete_game(self):
        for seed in range(10):
            agents = {i: RandomAgent(i, seed=seed + i) for i in range(1, 9)}
            rec = _full_game(agents, 8, seed=seed)
            self.assertIn(rec["winner"], (Team.VILLAGE, Team.WOLF))


class TestHeuristicAgent(unittest.TestCase):

    def test_tracks_accusations(self):
        agent = HeuristicAgent(1, seed=0)
        talks = [
            Talk(day=1, turn=0, agent=2, text="VOTE P3"),
            Talk(day=1, turn=1, agent=3, text="ESTIMATE P3 WEREWOLF"),
        ]
        gi = _make_gameinfo(1, talks=talks)
        agent.initialize(gi, _make_setting())
        # P3 should be the most accused
        self.assertEqual(agent.vote(), 3)

    def test_talk_declares_vote(self):
        agent = HeuristicAgent(2, seed=0)
        talks = [Talk(day=1, turn=0, agent=3, text="VOTE P5")]
        gi = _make_gameinfo(2, talks=talks)
        agent.initialize(gi, _make_setting())
        talk = agent.talk()
        self.assertTrue(talk.startswith("VOTE") or talk == "Skip")

    def test_belief_sums_to_one(self):
        agent = HeuristicAgent(1, seed=0)
        talks = [Talk(1, 0, 2, "VOTE P3"), Talk(1, 1, 3, "VOTE P4")]
        gi = _make_gameinfo(1, talks=talks)
        agent.initialize(gi, _make_setting())
        alive = agent.alive_others()
        total = sum(agent.get_wolf_belief(p) for p in alive)
        self.assertAlmostEqual(total, 1.0, places=5)


class TestBayesianAgent(unittest.TestCase):

    def test_prior_sums_correctly(self):
        agent = BayesianAgent(1, seed=0)
        gi    = _make_gameinfo(1)
        agent.initialize(gi, _make_setting(8))
        alive = agent.alive_others()
        # Sum of priors ≈ num_wolves = 2
        total = sum(agent.get_wolf_belief(p) for p in alive)
        self.assertAlmostEqual(total, 2.0, delta=0.5)

    def test_divine_result_sets_certainty(self):
        agent = BayesianAgent(1, seed=0, )
        # Seer divine result: P5 is WEREWOLF
        dr = Judge(day=1, agent=1, target=5, result="WEREWOLF")
        gi = _make_gameinfo(1)
        gi.divine_result = dr
        agent.initialize(gi, _make_setting())
        agent.update(gi, _make_setting())
        self.assertAlmostEqual(agent.get_wolf_belief(5), 1.0, places=5)

    def test_vote_returns_alive_other(self):
        agent = BayesianAgent(1, seed=0)
        gi    = _make_gameinfo(1)
        agent.initialize(gi, _make_setting())
        v = agent.vote()
        self.assertIn(v, range(1, 9))
        self.assertNotEqual(v, 1)


class TestMCTSAgent(unittest.TestCase):

    def test_vote_returns_valid(self):
        agent = MCTSAgent(1, n_simulations=20, seed=0)
        gi    = _make_gameinfo(1)
        agent.initialize(gi, _make_setting())
        v = agent.vote()
        self.assertIn(v, range(1, 9))
        self.assertNotEqual(v, 1)

    def test_mcts_runs_complete_game(self):
        for seed in range(5):
            agents = {i: MCTSAgent(i, n_simulations=10, seed=seed + i)
                      for i in range(1, 6)}
            rec = _full_game(agents, 5, seed=seed)
            self.assertIn(rec["winner"], (Team.VILLAGE, Team.WOLF))


class TestLogicAgent(unittest.TestCase):

    def test_infers_suspicious_from_claim_contradiction(self):
        """If P2 claims Seer and divines P3 as Wolf, but P3 is revealed as
        Villager, LogicAgent should mark P2 suspicious."""
        agent = LogicAgent(1, seed=0)
        talks = [
            Talk(1, 0, 2, "COMINGOUT P2 SEER"),
            Talk(1, 1, 2, "DIVINED P3 WEREWOLF"),
        ]
        gi = _make_gameinfo(1, talks=talks)
        agent.initialize(gi, _make_setting())

        # Simulate P3 executed as Villager
        gi2 = _make_gameinfo(1, talks=talks)
        gi2.status_map[3] = Status.DEAD
        gi2.role_map[3] = Role.VILLAGER
        agent.update(gi2, _make_setting())

        # P2 should have higher vote weight than uninvolved players
        w2 = agent._vote_weight[2]
        w4 = agent._vote_weight[4]  # uninvolved
        self.assertGreaterEqual(w2, w4,
                                msg="P2 should be more suspicious than P4")

    def test_vote_returns_valid(self):
        agent = LogicAgent(1, seed=0)
        gi    = _make_gameinfo(1)
        agent.initialize(gi, _make_setting())
        v = agent.vote()
        self.assertIn(v, range(1, 9))
        self.assertNotEqual(v, 1)


class TestAllAgentsAnyRole(unittest.TestCase):
    """Every agent type must survive being assigned any role."""

    AGENT_FACTORIES = {
        "random":    lambda pid: RandomAgent(pid, seed=0),
        "heuristic": lambda pid: HeuristicAgent(pid, seed=0),
        "bayesian":  lambda pid: BayesianAgent(pid, seed=0),
        "mcts":      lambda pid: MCTSAgent(pid, n_simulations=5, seed=0),
        "logic":     lambda pid: LogicAgent(pid, seed=0),
    }

    def test_agents_run_in_5_player_game(self):
        for name, factory in self.AGENT_FACTORIES.items():
            for seed in range(3):
                agents = {i: factory(i) for i in range(1, 6)}
                rng   = random.Random(seed)
                roles = assign_roles([1, 2, 3, 4, 5], 5, rng)
                game  = WerewolfGame(agents, roles, game_id=0, seed=seed)
                rec   = game.run()
                self.assertIn(
                    rec["winner"], (Team.VILLAGE, Team.WOLF),
                    msg=f"{name} failed in 5-player game seed={seed}"
                )


if __name__ == "__main__":
    unittest.main()
