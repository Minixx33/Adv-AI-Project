"""
runner/game_runner.py

GameRunner — orchestrates N games, builds agent compositions, runs each game,
calls all three logging components, and collects summary statistics.

Hierarchy:
  GameRunner
    ├── build_agents(composition, n_players, suspicion_agent_type, config)
    │       → list[AbstractAgent]   plus the EnhancedAgent if requested
    ├── run_games(n_games, config)
    │       → list[game_record], summary_stats_dict
    └── _run_one(game_id, agents, roles, config)
          → game_record

Agent composition (config["agent_composition"]):
  Keys: "random", "heuristic", "bayesian", "mcts", "logic_based",
        "bayesian_with_suspicion", "heuristic_with_suspicion", "mcts_with_suspicion"
  Values: count of that agent type.

If suspicion_agent_type != "none", exactly ONE suspicion-enhanced agent
is included (added automatically if not in composition).

Per game:
  1. Assign roles via role_assigner.assign_roles (with suspicion-not-wolf constraint)
  2. Run WerewolfGame
  3. Call ConversationLogger.write()
  4. If suspicion agent present: call SuspicionTracer.write() and EvolutionLogger.write()
  5. Save game_summary.json
  6. Accumulate summary row for summary.csv
"""

import json
import os
import random
import csv
import sys
from typing import Optional

# Ensure Model_V2/ is on sys.path so sibling packages are importable
# regardless of the working directory or how this file is invoked.
_MODEL_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_V2 not in sys.path:
    sys.path.insert(0, _MODEL_V2)

from werewolf.game        import WerewolfGame
from werewolf.role_assigner import assign_roles
from werewolf.const       import ROLE_CONFIG, Team, Role
from werewolf.agents.random_agent   import RandomAgent
from werewolf.agents.heuristic_agent import HeuristicAgent
from werewolf.agents.bayesian_agent  import BayesianAgent
from werewolf.agents.mcts_agent     import MCTSAgent
from werewolf.agents.logic_agent    import LogicAgent
from suspicion.enhanced_agent       import EnhancedAgent

from runner.xlsx_exporter    import XlsxExporter
from runner.evolution_logger import EvolutionLogger
from suspicion.online_learner import OnlineLearner


# Short labels used as CSV column prefixes for per-type vote stats
_TYPE_SHORT = {
    "random":                   "rnd",
    "heuristic":                "heur",
    "bayesian":                 "bay",
    "mcts":                     "mcts",
    "logic_based":              "logic",
    "heuristic_with_suspicion": "heur_susp",
    "bayesian_with_suspicion":  "bay_susp",
    "mcts_with_suspicion":      "mcts_susp",
}

_SUMMARY_FIELDS = [
    "game_id", "n_players", "winner", "n_days",
    "suspicion_agent_id", "suspicion_agent_role",
    "suspicion_agent_survived", "suspicion_agent_won",
    "correct_wolf_votes", "total_votes", "false_accusations",
] + [f"{s}_correct_pct" for s in _TYPE_SHORT.values()]

_AGENT_FACTORIES = {
    "random":     lambda pid, seed: RandomAgent(pid, seed=seed),
    "heuristic":  lambda pid, seed: HeuristicAgent(pid, seed=seed),
    "bayesian":   lambda pid, seed: BayesianAgent(pid, seed=seed),
    "mcts":       lambda pid, seed: MCTSAgent(pid, seed=seed),
    "logic_based": lambda pid, seed: LogicAgent(pid, seed=seed),
}

_AGENT_DISPLAY_NAMES = {
    "random":                   "Random",
    "heuristic":                "Heuristic",
    "bayesian":                 "Bayesian",
    "mcts":                     "MCTS",
    "logic_based":              "Logic",
    "bayesian_with_suspicion":  "Bayesian+Suspicion",
    "heuristic_with_suspicion": "Heuristic+Suspicion",
    "mcts_with_suspicion":      "MCTS+Suspicion",
}


class GameRunner:
    """
    High-level orchestrator for running N Werewolf games.

    Parameters
    ----------
    config : dict
        Must contain at minimum:
          players            : int (5, 8, or 10)
          output_dir         : str
          suspicion_agent_type : str ("heuristic"|"bayesian"|"mcts"|"none")
        Optional:
          agent_composition  : dict (agent type → count)
          suspicion_weights  : {"a1"..."a5", "w1", "w2"}
          mcts_simulations   : int
          mcts_depth         : int
          seed               : int | None
          verbose            : bool
    """

    def __init__(self, config: dict):
        self._config = config
        self._rng    = random.Random(config.get("seed"))
        self._xlsx_exporter = XlsxExporter()
        self._evol_logger   = EvolutionLogger()
        self._learner: Optional[OnlineLearner] = None

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run_games(self, n_games: int) -> tuple:
        """
        Run n_games games.
        Returns (list[game_record], summary_rows_list).
        """
        output_dir = self._config.get("output_dir", "./results")
        os.makedirs(output_dir, exist_ok=True)

        # Print agent composition before starting
        n_players = self._config.get("players", 8)
        susp_type = self._config.get("suspicion_agent_type", "bayesian")
        comp = self._config.get("agent_composition") or \
               self._default_composition(n_players, susp_type)
        comp_str = "  |  ".join(
            f"{cnt}x {_AGENT_DISPLAY_NAMES.get(t, t)}"
            for t, cnt in comp.items() if cnt > 0
        )
        print(f"Agent composition:  {comp_str}")

        # Initialise online learner if any suspicion agents are present
        has_suspicion = (
            susp_type != "none"
            or any("_with_suspicion" in k for k in comp)
        )
        if has_suspicion:
            self._learner = OnlineLearner(
                initial_weights=self._config.get("suspicion_weights"),
                lr=self._config.get("learning_rate", 0.01),
            )

        summary_rows: list = []
        records:      list = []

        print_every = max(1, n_games // 100)  # ~1% intervals

        for game_id in range(n_games):
            record, summary_row = self._run_one(game_id)
            records.append(record)
            summary_rows.append(summary_row)

            completed = game_id + 1
            if completed % print_every == 0 or completed == n_games:
                pct = completed / n_games * 100
                print(f"  {completed}/{n_games} games completed ({pct:.0f}%)",
                      flush=True)
            elif self._config.get("verbose") and game_id % 10 == 0:
                print(f"  Completed game {game_id}/{n_games}")

        self._write_summary(summary_rows, output_dir)
        self._write_config_echo(output_dir)

        if self._learner and self._learner.history:
            hist_path = self._learner.write_history(output_dir)
            print(f"Weight history saved -> {hist_path}")

        return records, summary_rows

    # ------------------------------------------------------------------ #
    #  Single-game runner                                                  #
    # ------------------------------------------------------------------ #

    def _run_one(self, game_id: int) -> tuple:
        n_players  = self._config["players"]
        output_dir = self._config.get("output_dir", "./results")
        seed       = self._rng.randint(0, 2**31) if self._config.get("seed") is None \
                     else self._config["seed"] + game_id

        # 1. Build agents and collect all suspicion slots
        agents_list, susp_pids, no_wolf_pids, pid_to_type = self._build_agents(
            n_players, game_id, seed
        )
        player_ids  = [a.agent_id for a in agents_list]
        agents_dict = {a.agent_id: a for a in agents_list}

        # 2. Assign roles (no-wolf constraint: suspicion agents + logic agents)
        roles = assign_roles(
            player_ids=player_ids,
            n_players=n_players,
            rng=random.Random(seed),
            suspicion_player_ids=no_wolf_pids,
        )

        # 3. Give each enhanced agent the ground-truth role map (for trace logging)
        for pid in susp_pids:
            enh = agents_dict.get(pid)
            if isinstance(enh, EnhancedAgent):
                enh.set_true_roles(roles)

        # 3b. Assign wolf talk strategies (if configured)
        wolf_cfg = self._config.get("wolf_strategies", {})
        if wolf_cfg:
            self._assign_wolf_strategies(agents_dict, roles, seed, wolf_cfg)

        # 4. Run the game
        game = WerewolfGame(
            agents=agents_dict,
            roles=roles,
            game_id=game_id,
            seed=seed,
            max_talk_rounds=self._config.get("max_talk_rounds", 5),
            verbose=self._config.get("verbose", False),
            suspicion_agent_ids=susp_pids,
        )
        record = game.run()

        # 5. Logging
        game_dir = os.path.join(output_dir, f"game_{game_id:04d}")
        os.makedirs(game_dir, exist_ok=True)

        # Gather per-agent suspicion data; write one evolution CSV per agent
        susp_agents_data: list = []
        all_evo_rows:     list = []
        for pid in susp_pids:
            enh = agents_dict.get(pid)
            if isinstance(enh, EnhancedAgent):
                evo_rows = enh.get_evolution_rows()
                all_evo_rows.extend(evo_rows)
                susp_agents_data.append({
                    "agent_id":        pid,
                    "agent_name":      enh.name,
                    "trace_snapshots": enh.get_trace_snapshots(),
                    "decision_weights": enh._decision_weights,
                })
                self._evol_logger.write(
                    evolution_rows=evo_rows,
                    game_id=game_id,
                    output_dir=output_dir,
                )

        # Online learning: update suspicion weights from this game's data
        if self._learner is not None and all_evo_rows:
            new_weights = self._learner.update(game_id, all_evo_rows)
            self._config["suspicion_weights"] = new_weights

        self._xlsx_exporter.write(
            game_record=record,
            suspicion_agents=susp_agents_data,
            true_roles=roles,
            output_dir=output_dir,
        )

        # 7. Per-game JSON summary
        first_susp = susp_pids[0] if susp_pids else None
        self._write_game_summary(record, roles, first_susp, game_dir)

        # 8. Build summary row
        summary_row = self._build_summary_row(record, roles, first_susp, pid_to_type)
        return record, summary_row

    # ------------------------------------------------------------------ #
    #  Agent construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_agents(self, n_players: int, game_id: int, seed: int) -> tuple:
        """
        Build the agent list from composition config.
        Returns (agents_list, susp_pids: list[int], pid_to_type: dict).
        Multiple *_with_suspicion entries are all collected.
        """
        susp_type   = self._config.get("suspicion_agent_type", "bayesian")
        composition = self._config.get("agent_composition", None)

        if composition is None:
            composition = self._default_composition(n_players, susp_type)

        # Validate total
        total = sum(composition.values())
        if total != n_players:
            raise ValueError(
                f"agent_composition sums to {total}, expected {n_players}. "
                f"Composition: {composition}"
            )

        agents:        list = []
        susp_pids:     list = []   # suspicion-enhanced only (game notifications)
        no_wolf_pids:  list = []   # suspicion + logic (role-assignment constraint)
        pid_to_type:   dict = {}
        pid = 1

        susp_keys = {
            "bayesian_with_suspicion", "heuristic_with_suspicion",
            "mcts_with_suspicion",
        }

        for agent_type, count in composition.items():
            for _ in range(count):
                agent_seed = seed + pid * 1000 if seed is not None else None

                if agent_type in susp_keys:
                    base_type    = agent_type.replace("_with_suspicion", "")
                    base_factory = _AGENT_FACTORIES.get(base_type)
                    if base_factory is None:
                        raise ValueError(f"Unknown base agent type: {base_type}")
                    base = base_factory(pid, agent_seed)
                    susp_weights = self._config.get("suspicion_weights", {})
                    det_w = {k: susp_weights[k] for k in ("a1","a2","a3","a4","a5")
                             if k in susp_weights} or None
                    dec_w = {k: susp_weights[k] for k in ("w1","w2")
                             if k in susp_weights} or None
                    agent = EnhancedAgent(base, det_w, dec_w)
                    susp_pids.append(pid)
                    no_wolf_pids.append(pid)
                elif agent_type == "none":
                    pass  # skip
                elif agent_type in ("random", "heuristic", "bayesian",
                                     "mcts", "logic_based"):
                    factory = _AGENT_FACTORIES[agent_type]
                    agent   = factory(pid, agent_seed)
                    if agent_type == "mcts":
                        agent.n_simulations = self._config.get(
                            "mcts_simulations", 500)
                        agent.max_depth     = self._config.get(
                            "mcts_depth", 5)
                    if agent_type == "logic_based":
                        no_wolf_pids.append(pid)
                else:
                    raise ValueError(f"Unknown agent type: '{agent_type}'")

                pid_to_type[pid] = agent_type
                agents.append(agent)
                pid += 1

        # If suspicion requested but none in composition, auto-insert one
        if susp_type != "none" and not susp_pids and len(agents) < n_players:
            agent_seed   = seed + pid * 1000 if seed is not None else None
            base_factory = _AGENT_FACTORIES.get(susp_type, _AGENT_FACTORIES["bayesian"])
            base = base_factory(pid, agent_seed)
            susp_weights = self._config.get("suspicion_weights", {})
            det_w = {k: susp_weights[k] for k in ("a1","a2","a3","a4","a5")
                     if k in susp_weights} or None
            dec_w = {k: susp_weights[k] for k in ("w1","w2")
                     if k in susp_weights} or None
            agent = EnhancedAgent(base, det_w, dec_w)
            auto_type = f"{susp_type}_with_suspicion"
            susp_pids.append(pid)
            no_wolf_pids.append(pid)
            pid_to_type[pid] = auto_type
            agents.append(agent)

        return agents, susp_pids, no_wolf_pids, pid_to_type

    def _assign_wolf_strategies(
        self,
        agents: dict,
        roles: dict,
        seed: int,
        wolf_cfg: dict,
    ) -> None:
        """
        For each wolf agent, randomly pick a talk strategy based on configured
        probabilities. Strategies are mutually exclusive per agent.
        """
        from werewolf.const import Role as WRole
        rng = random.Random(seed + 77777)  # deterministic but independent from game rng
        p_bus     = wolf_cfg.get("bus_driver_probability", 0.2)
        p_false   = wolf_cfg.get("false_claimer_probability", 0.5)

        for pid, agent in agents.items():
            if roles.get(pid) != WRole.WEREWOLF:
                continue
            r = rng.random()
            if r < p_bus:
                agent.set_wolf_strategy("bus_driver")
            elif r < p_bus + p_false:
                agent.set_wolf_strategy("false_claimer")
            # else: no special strategy — default talk behaviour

    def _default_composition(self, n_players: int, susp_type: str) -> dict:
        """
        If no composition given, build a balanced mix.
        Always includes one suspicion agent if susp_type != 'none'.
        """
        if susp_type != "none":
            n_other = n_players - 1
            n_random   = max(1, n_other // 3)
            n_heuristic = max(1, n_other // 3)
            n_bayesian  = n_other - n_random - n_heuristic
            return {
                "random":    n_random,
                "heuristic": n_heuristic,
                "bayesian":  n_bayesian,
                f"{susp_type}_with_suspicion": 1,
            }
        else:
            n_random   = max(1, n_players // 3)
            n_heuristic = max(1, n_players // 3)
            n_bayesian  = n_players - n_random - n_heuristic
            return {
                "random":    n_random,
                "heuristic": n_heuristic,
                "bayesian":  n_bayesian,
            }

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_summary_row(
        self,
        record:      dict,
        roles:       dict,
        susp_pid:    Optional[int],
        pid_to_type: dict,
    ) -> dict:
        winner = record["winner"]

        # Overall vote counters + per-agent-type buckets
        total_votes        = 0
        correct_wolf_votes = 0
        false_accusations  = 0
        type_total:   dict = {}   # agent_type -> votes cast
        type_correct: dict = {}   # agent_type -> correct wolf votes

        for ev in record["events"]:
            if ev.get("type") == "vote":
                total_votes += 1
                voter       = ev["agent"]
                target_role = roles.get(ev["target"], Role.VILLAGER)
                is_wolf     = target_role in (Role.WEREWOLF,)

                atype = pid_to_type.get(voter, "random")
                type_total[atype]   = type_total.get(atype, 0) + 1
                if is_wolf:
                    correct_wolf_votes += 1
                    type_correct[atype] = type_correct.get(atype, 0) + 1
                else:
                    false_accusations += 1

        susp_survived = None
        susp_won      = None
        susp_role     = None
        if susp_pid is not None:
            susp_role = roles.get(susp_pid, Role.VILLAGER).value
            from werewolf.const import ROLE_TO_TEAM
            susp_team = ROLE_TO_TEAM.get(roles.get(susp_pid, Role.VILLAGER))
            susp_survived = record["status"].get(susp_pid) == "ALIVE"
            susp_won      = (susp_team == winner)

        row = {
            "game_id":               record["game_id"],
            "n_players":             record["n_players"],
            "winner":                winner.value,
            "n_days":                record["n_days"],
            "suspicion_agent_id":    f"P{susp_pid}" if susp_pid else "",
            "suspicion_agent_role":  susp_role or "",
            "suspicion_agent_survived": susp_survived,
            "suspicion_agent_won":      susp_won,
            "correct_wolf_votes":    correct_wolf_votes,
            "total_votes":           total_votes,
            "false_accusations":     false_accusations,
        }

        # Per-type correct vote percentage (None when that type cast no votes)
        for atype, short in _TYPE_SHORT.items():
            n = type_total.get(atype, 0)
            c = type_correct.get(atype, 0)
            row[f"{short}_correct_pct"] = round(c / n, 4) if n > 0 else None

        return row

    def _write_summary(self, rows: list, output_dir: str) -> None:
        path = os.path.join(output_dir, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_config_echo(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "config_used.json")
        serialisable = {
            k: v for k, v in self._config.items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

    def _write_game_summary(
        self, record: dict, roles: dict,
        susp_pid: Optional[int], game_dir: str
    ) -> None:
        summary = {
            "game_id":   record["game_id"],
            "n_players": record["n_players"],
            "winner":    record["winner"].value,
            "n_days":    record["n_days"],
            "roles":     record["roles"],
            "agents":    record["agent_names"],
            "final_status": record["status"],
            "suspicion_agent_id":   susp_pid,
            "suspicion_agent_role": record.get("suspicion_agent_role"),
        }
        path = os.path.join(game_dir, "game_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
