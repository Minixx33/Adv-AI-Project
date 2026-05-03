"""
werewolf/agents/mcts_agent.py

MCTSAgent — Monte Carlo Tree Search baseline.

Hierarchy:
  AbstractAgent
    └── MCTSAgent
          └── _MCTSNode  — tree node (internal class)

Algorithm (spec §1):
  For each vote/attack/divine/guard decision:
    1. Build a lightweight game-state snapshot from current GameInfo.
    2. From the root (current state), run N_SIMULATIONS rollouts.
       Each rollout:
         a. Sample a possible role assignment for unknown players.
         b. Play out the game with random agents for both sides.
         c. Score: +1 if the deciding agent's team wins, 0 otherwise.
    3. UCB1 selection during tree traversal:
         UCB1(node) = mean(value) + C * sqrt(ln(N_parent) / N_node)
    4. Return the action with the highest mean value.

Configurable (via constructor or keyword args):
  n_simulations : int  — rollouts per decision (default 500)
  max_depth     : int  — playout depth cap (default 5)
  ucb_c         : float — exploration constant (default √2)

Talk:
  After each round of simulations, announces "VOTE P{target}" to the group.

Compatible with the suspicion module (§2):
  get_wolf_belief() returns the normalised simulation-vote fraction per player.
"""

import math
import random
import re as _re
from collections import defaultdict, Counter
from typing import Optional, Tuple

from ..agent import AbstractAgent
from ..gameinfo import GameInfo, GameSetting
from ..const import Role, Status, Team, WOLF_ROLES, ROLE_CONFIG, ROLE_TO_SPECIES


class _MCTSNode:
    """Single node in the MCTS tree. Each node represents one possible action."""

    __slots__ = ("action", "parent", "children", "wins", "visits", "untried_actions")

    def __init__(self, action=None, parent=None, untried_actions=None):
        self.action          = action        # the action that led here
        self.parent          = parent
        self.children: list  = []
        self.wins:   float   = 0.0
        self.visits: int     = 0
        self.untried_actions = list(untried_actions or [])

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return math.inf
        return (self.wins / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c: float) -> "_MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return not self.children and self.is_fully_expanded()


class MCTSAgent(AbstractAgent):
    """
    MCTS-based decision agent.  Each decision triggers up to n_simulations
    rollouts from the current observable state.
    """

    def __init__(
        self,
        agent_id:      int,
        n_simulations: int   = 500,
        max_depth:     int   = 5,
        ucb_c:         float = math.sqrt(2),
        seed:          int   = None,
    ):
        super().__init__(agent_id, name="MCTSAgent", seed=seed)
        self.n_simulations = n_simulations
        self.max_depth     = max_depth
        self.ucb_c         = ucb_c

        self._sim_votes: dict  = defaultdict(int)   # {pid: wins} across simulations
        self._last_decision: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().initialize(game_info, game_setting)
        self._sim_votes = defaultdict(int)
        self._last_decision = None

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().update(game_info, game_setting)

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_round_tokens(self) -> list:
        target = self._decide_vote_target()
        if target is None:
            return ["Skip"]
        tokens = [f"ESTIMATE P{target} WEREWOLF", f"VOTE P{target}"]
        dominant_target, accuser = self._dominant_accuser()
        if accuser is not None:
            if dominant_target == target:
                tokens.append(f"AGREE P{accuser}")
            else:
                tokens.append(f"DISAGREE P{accuser}")
        tokens.append("Over")
        return tokens

    def _dominant_accuser(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (most-accused player, pid of most-recent accuser) from talk_list."""
        if not self.game_info:
            return None, None
        counts: Counter = Counter()
        last_accuser: dict = {}
        for talk in self.game_info.talk_list:
            if talk.agent == self.agent_id:
                continue
            for m in _re.finditer(r"\b(?:ATTACK|VOTE)\s+P(\d+)", talk.text, _re.IGNORECASE):
                pid = int(m.group(1))
                counts[pid] += 1
                last_accuser[pid] = talk.agent
        if not counts:
            return None, None
        dominant = counts.most_common(1)[0][0]
        return dominant, last_accuser.get(dominant)

    def vote(self) -> int:
        target = self._decide_vote_target()
        return target if target is not None else self._rng.choice(
            self.alive_others() or [self.agent_id]
        )

    def attack(self) -> int:
        candidates = [
            p for p in self.game_info.alive_agents
            if p != self.agent_id
            and self.game_info.role_map.get(p) != Role.WEREWOLF
        ]
        if not candidates:
            return self._random_non_wolf_target()
        return self._mcts_best_action(candidates)

    def divine(self) -> int:
        others = self.alive_others()
        return self._mcts_best_action(others) if others else self.agent_id

    def guard(self) -> int:
        others = self.alive_others()
        return self._rng.choice(others) if others else self.agent_id

    # ------------------------------------------------------------------ #
    #  Belief interface                                                    #
    # ------------------------------------------------------------------ #

    def get_wolf_belief(self, player_id: int) -> float:
        """
        Fraction of simulations in which player_id was the chosen vote target
        (used as proxy for wolf probability by the SuspicionModule).
        """
        total = sum(self._sim_votes.values())
        if total == 0:
            n = self.game_setting.player_num if self.game_setting else 8
            n_wolves = (self.game_setting.role_num_map.get(Role.WEREWOLF, 0)
                        if self.game_setting else 2)
            return n_wolves / max(n - 1, 1)
        return self._sim_votes.get(player_id, 0) / total

    # ------------------------------------------------------------------ #
    #  MCTS core                                                           #
    # ------------------------------------------------------------------ #

    def _decide_vote_target(self) -> Optional[int]:
        candidates = self.alive_others()
        if not candidates:
            return None
        best = self._mcts_best_action(candidates)
        self._last_decision = best
        return best

    def _mcts_best_action(self, actions: list) -> int:
        """Run MCTS over the given actions; return the action with most wins."""
        if len(actions) == 1:
            return actions[0]

        sim_rng = random.Random(self._rng.random())
        root = _MCTSNode(untried_actions=list(actions))
        win_counts  = defaultdict(int)
        visit_counts = defaultdict(int)

        for _ in range(self.n_simulations):
            # Selection + expansion
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.ucb_c)

            # Expand
            if node.untried_actions:
                action = sim_rng.choice(node.untried_actions)
                node.untried_actions.remove(action)
                child = _MCTSNode(action=action, parent=node,
                                  untried_actions=[])
                node.children.append(child)
                node = child

            action = node.action if node.action is not None else sim_rng.choice(actions)

            # Simulate (rollout)
            result = self._simulate(action, sim_rng)

            # Backpropagate
            n = node
            while n is not None:
                n.visits += 1
                n.wins   += result
                n = n.parent

            win_counts[action]   += result
            visit_counts[action] += 1

        # Store simulation vote counts for get_wolf_belief()
        self._sim_votes = win_counts

        # Return action with highest win rate
        if not visit_counts:
            return sim_rng.choice(actions)
        return max(actions, key=lambda a: win_counts[a] / max(visit_counts[a], 1))

    def _simulate(self, chosen_action: int, rng: random.Random) -> float:
        """
        Roll out a game from the current state using `chosen_action` as our
        first move, then uniform-random agents for all subsequent moves.
        Returns 1.0 if our team wins, 0.0 otherwise.
        """
        if not self.game_info or not self.game_setting:
            return 0.5

        # Sample a plausible world
        world = self._sample_world(chosen_action, rng)
        if world is None:
            return 0.5

        my_team = self._get_my_team(world)
        status  = {p: s for p, s in self.game_info.status_map.items()}
        roles   = dict(world)
        day     = self.game_info.day

        # Simulate initial action
        self._apply_sim_action(chosen_action, roles, status, rng)

        # Rollout up to max_depth steps
        for depth in range(self.max_depth):
            alive = [p for p, s in status.items() if s == Status.ALIVE]
            alive_wolves  = [p for p in alive if roles[p] in WOLF_ROLES]
            alive_others  = [p for p in alive if roles[p] not in WOLF_ROLES]

            if not alive_wolves:
                return 1.0 if my_team == Team.VILLAGE else 0.0
            if len(alive_wolves) >= len(alive_others):
                return 1.0 if my_team == Team.WOLF else 0.0

            # Random village vote — eliminate random player
            target = rng.choice(alive)
            status[target] = Status.DEAD

            alive  = [p for p, s in status.items() if s == Status.ALIVE]
            alive_wolves = [p for p in alive if roles[p] in WOLF_ROLES]
            alive_others = [p for p in alive if roles[p] not in WOLF_ROLES]

            if not alive_wolves:
                return 1.0 if my_team == Team.VILLAGE else 0.0
            if len(alive_wolves) >= len(alive_others):
                return 1.0 if my_team == Team.WOLF else 0.0

            # Random wolf attack
            non_wolf_alive = [p for p in alive if roles[p] not in WOLF_ROLES]
            if non_wolf_alive:
                attack_target = rng.choice(non_wolf_alive)
                status[attack_target] = Status.DEAD

        # Evaluate end-of-playout state
        alive = [p for p, s in status.items() if s == Status.ALIVE]
        alive_wolves = [p for p in alive if roles[p] in WOLF_ROLES]
        alive_others = [p for p in alive if roles[p] not in WOLF_ROLES]

        if not alive_wolves:
            winner = Team.VILLAGE
        elif len(alive_wolves) >= len(alive_others):
            winner = Team.WOLF
        else:
            winner = Team.VILLAGE  # partial result; assume village leads

        return 1.0 if winner == my_team else 0.0

    def _sample_world(self, chosen_action: int, rng: random.Random) -> Optional[dict]:
        """
        Sample a consistent possible world (role assignment for unknowns).
        Known roles (own + teammates + revealed dead) are fixed.
        """
        if not self.game_info or not self.game_setting:
            return None

        known   = dict(self.game_info.role_map)
        unknown = [p for p in self.game_info.status_map if p not in known]

        # Count remaining roles to distribute
        dist = dict(self.game_setting.role_num_map)
        for role in known.values():
            dist[role] = max(0, dist.get(role, 0) - 1)

        pool = [r for r, cnt in dist.items() for _ in range(cnt)]
        if len(pool) != len(unknown):
            return None   # inconsistent — skip this sample

        rng.shuffle(pool)
        world = dict(known)
        for pid, role in zip(unknown, pool):
            world[pid] = role
        return world

    def _get_my_team(self, world: dict) -> Team:
        from ..const import ROLE_TO_TEAM
        my_role = world.get(self.agent_id, Role.VILLAGER)
        return ROLE_TO_TEAM.get(my_role, Team.VILLAGE)

    def _apply_sim_action(self, action: int, roles: dict,
                          status: dict, rng: random.Random) -> None:
        """Apply the chosen action to the simulated world."""
        if status.get(action) == Status.ALIVE:
            status[action] = Status.DEAD
