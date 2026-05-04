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

from werewolf.game         import WerewolfGame
from werewolf.role_assigner import assign_roles
from werewolf.const        import ROLE_CONFIG, Team, Role
from werewolf.agents.random_agent    import RandomAgent
from werewolf.agents.heuristic_agent import HeuristicAgent
from werewolf.agents.bayesian_agent  import BayesianAgent
from werewolf.agents.mcts_agent      import MCTSAgent
from werewolf.agents.logic_agent     import LogicAgent
from suspicion.enhanced_agent        import EnhancedAgent

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

_SUMMARY_FIELDS = (
    # Outcome
    ["game_id", "n_players", "winner", "village_win", "n_days",
    # Suspicion agents
    "n_suspicion_agents",
    "suspicion_agent_id", "suspicion_agent_role",
    "suspicion_agent_survived", "suspicion_agent_won",
    "n_susp_agents_survived", "n_susp_agents_won",
    # Vote metrics (plan sec 11)
    "total_votes", "correct_wolf_votes", "false_accusations",
    "wolf_vote_precision", "wolf_vote_recall", "false_accusation_rate",
    # Wolf elimination
    "total_wolves", "wolves_executed", "day_first_wolf_executed",]
    # Per-agent-type vote counts + accuracy
    + [f"{s}_votes"       for s in _TYPE_SHORT.values()]
    + [f"{s}_correct_pct" for s in _TYPE_SHORT.values()]
    # Suspicion calibration
    + ["mean_sigma_wolves", "mean_sigma_nonwolves", "sigma_gap",
       "mean_combined_wolves", "mean_combined_nonwolves", "combined_gap",
    # Detector means - overall
    "mean_d1", "mean_d2", "mean_d3", "mean_d4", "mean_d5",
    # Detector means - wolves vs non-wolves (RQ2)
    "mean_d1_wolves", "mean_d1_nonwolves",
    "mean_d2_wolves", "mean_d2_nonwolves",
    "mean_d3_wolves", "mean_d3_nonwolves",
    "mean_d4_wolves", "mean_d4_nonwolves",
    "mean_d5_wolves", "mean_d5_nonwolves",]
)

_AGENT_FACTORIES = {
    "random":      lambda pid, seed: RandomAgent(pid, seed=seed),
    "heuristic":   lambda pid, seed: HeuristicAgent(pid, seed=seed),
    "bayesian":    lambda pid, seed: BayesianAgent(pid, seed=seed),
    "mcts":        lambda pid, seed: MCTSAgent(pid, seed=seed),
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


class _Tee:
    """Mirrors stdout writes to a log file simultaneously."""
    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._log    = open(log_path, 'w', encoding='utf-8', buffering=1)
        sys.stdout   = self

    def write(self, text):
        self._stdout.write(text)
        self._log.write(text)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()


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
        self._config        = config
        self._rng           = random.Random(config.get("seed"))
        self._xlsx_exporter = XlsxExporter()
        self._evol_logger   = EvolutionLogger()
        self._learner: Optional[OnlineLearner] = None

    # --- Public -----------------------------------------------------------

    def run_games(self, n_games: int) -> tuple:
        """Run n_games games. Returns (list[game_record], summary_rows_list)."""
        output_dir = self._config.get("output_dir", "./results")
        os.makedirs(output_dir, exist_ok=True)

        import datetime
        log_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        tee = _Tee(os.path.join(output_dir, log_name))
        try:
            return self._run_games_inner(n_games, output_dir)
        finally:
            tee.close()

    def _run_games_inner(self, n_games: int, output_dir: str) -> tuple:
        n_players = self._config.get("players", 8)
        susp_type = self._config.get("suspicion_agent_type", "bayesian")
        comp = self._config.get("agent_composition") or \
               self._default_composition(n_players, susp_type)
        comp_str = "  |  ".join(
            f"{cnt}x {_AGENT_DISPLAY_NAMES.get(t, t)}"
            for t, cnt in comp.items() if cnt > 0
        )
        print(f"Agent composition:  {comp_str}")

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
        print_every = max(1, n_games // 100)

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
        self._print_aggregate(summary_rows, output_dir)

        if self._learner and self._learner.history:
            hist_path = self._learner.write_history(output_dir)
            print(f"  Weight history -> {hist_path}")

        return records, summary_rows

    # --- Single-game runner -----------------------------------------------

    def _run_one(self, game_id: int) -> tuple:
        n_players  = self._config["players"]
        output_dir = self._config.get("output_dir", "./results")
        seed       = self._rng.randint(0, 2**31) if self._config.get("seed") is None \
                     else self._config["seed"] + game_id

        agents_list, susp_pids, no_wolf_pids, pid_to_type = self._build_agents(
            n_players, game_id, seed
        )
        player_ids  = [a.agent_id for a in agents_list]
        agents_dict = {a.agent_id: a for a in agents_list}

        roles = assign_roles(
            player_ids=player_ids,
            n_players=n_players,
            rng=random.Random(seed),
            suspicion_player_ids=no_wolf_pids,
        )

        for pid in susp_pids:
            enh = agents_dict.get(pid)
            if isinstance(enh, EnhancedAgent):
                enh.set_true_roles(roles)

        wolf_cfg = self._config.get("wolf_strategies", {})
        if wolf_cfg:
            self._assign_wolf_strategies(agents_dict, roles, seed, wolf_cfg)

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

        game_dir = os.path.join(output_dir, f"game_{game_id:04d}")
        os.makedirs(game_dir, exist_ok=True)

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

        if self._learner is not None and all_evo_rows:
            new_weights = self._learner.update(game_id, all_evo_rows)
            self._config["suspicion_weights"] = new_weights

        self._xlsx_exporter.write(
            game_record=record,
            suspicion_agents=susp_agents_data,
            true_roles=roles,
            output_dir=output_dir,
        )

        first_susp = susp_pids[0] if susp_pids else None
        self._write_game_summary(record, roles, first_susp, game_dir)

        summary_row = self._build_summary_row(
            record, roles, first_susp, pid_to_type,
            all_evo_rows=all_evo_rows, susp_pids=susp_pids,
        )
        return record, summary_row

    # --- Agent construction -----------------------------------------------

    def _build_agents(self, n_players: int, game_id: int, seed: int) -> tuple:
        susp_type   = self._config.get("suspicion_agent_type", "bayesian")
        composition = self._config.get("agent_composition", None)
        if composition is None:
            composition = self._default_composition(n_players, susp_type)

        total = sum(composition.values())
        if total != n_players:
            raise ValueError(
                f"agent_composition sums to {total}, expected {n_players}. "
                f"Composition: {composition}"
            )

        agents:       list = []
        susp_pids:    list = []
        no_wolf_pids: list = []
        pid_to_type:  dict = {}
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
                    pid += 1
                    continue
                elif agent_type in ("random", "heuristic", "bayesian",
                                    "mcts", "logic_based"):
                    factory = _AGENT_FACTORIES[agent_type]
                    agent   = factory(pid, agent_seed)
                    if agent_type == "mcts":
                        agent.n_simulations = self._config.get("mcts_simulations", 500)
                        agent.max_depth     = self._config.get("mcts_depth", 5)
                    if agent_type == "logic_based":
                        no_wolf_pids.append(pid)
                else:
                    raise ValueError(f"Unknown agent type: '{agent_type}'")

                pid_to_type[pid] = agent_type
                agents.append(agent)
                pid += 1

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
            susp_pids.append(pid)
            no_wolf_pids.append(pid)
            pid_to_type[pid] = f"{susp_type}_with_suspicion"
            agents.append(agent)

        return agents, susp_pids, no_wolf_pids, pid_to_type

    def _assign_wolf_strategies(self, agents, roles, seed, wolf_cfg):
        from werewolf.const import Role as WRole
        rng     = random.Random(seed + 77777)
        p_bus   = wolf_cfg.get("bus_driver_probability", 0.2)
        p_false = wolf_cfg.get("false_claimer_probability", 0.5)
        for pid, agent in agents.items():
            if roles.get(pid) != WRole.WEREWOLF:
                continue
            r = rng.random()
            if r < p_bus:
                agent.set_wolf_strategy("bus_driver")
            elif r < p_bus + p_false:
                agent.set_wolf_strategy("false_claimer")

    def _default_composition(self, n_players: int, susp_type: str) -> dict:
        if susp_type != "none":
            n_other     = n_players - 1
            n_random    = max(1, n_other // 3)
            n_heuristic = max(1, n_other // 3)
            n_bayesian  = n_other - n_random - n_heuristic
            return {
                "random":    n_random,
                "heuristic": n_heuristic,
                "bayesian":  n_bayesian,
                f"{susp_type}_with_suspicion": 1,
            }
        else:
            n_random    = max(1, n_players // 3)
            n_heuristic = max(1, n_players // 3)
            n_bayesian  = n_players - n_random - n_heuristic
            return {"random": n_random, "heuristic": n_heuristic, "bayesian": n_bayesian}

    # --- Summary row builder ----------------------------------------------

    def _build_summary_row(self, record, roles, susp_pid, pid_to_type,
                           all_evo_rows=None, susp_pids=None):
        from werewolf.const import ROLE_TO_TEAM
        winner       = record["winner"]
        all_evo_rows = all_evo_rows or []
        susp_pids    = susp_pids    or []

        # Vote counters
        total_votes = correct_wolf_votes = false_accusations = 0
        wolves_targeted: set = set()
        type_total:  dict = {}
        type_correct: dict = {}

        wolf_pids    = {pid for pid, role in roles.items() if role == Role.WEREWOLF}
        total_wolves = len(wolf_pids)

        for ev in record["events"]:
            if ev.get("type") != "vote":
                continue
            total_votes += 1
            voter  = ev["agent"]
            target = ev["target"]
            is_wolf = (roles.get(target, Role.VILLAGER) == Role.WEREWOLF)
            atype   = pid_to_type.get(voter, "random")
            type_total[atype] = type_total.get(atype, 0) + 1
            if is_wolf:
                correct_wolf_votes += 1
                wolves_targeted.add(target)
                type_correct[atype] = type_correct.get(atype, 0) + 1
            else:
                false_accusations += 1

        def _safe_div(a, b):
            return round(a / b, 4) if b > 0 else None

        wolf_vote_precision   = _safe_div(correct_wolf_votes, total_votes)
        wolf_vote_recall      = _safe_div(len(wolves_targeted), total_wolves)
        false_accusation_rate = _safe_div(false_accusations, total_votes)

        # Wolf elimination timing
        wolves_executed = 0
        day_first_wolf_executed = None
        for ev in record["events"]:
            if ev.get("type") == "execution" and ev.get("role") == "WEREWOLF":
                wolves_executed += 1
                if day_first_wolf_executed is None:
                    day_first_wolf_executed = ev["day"]

        # Suspicion agent fields
        susp_role = susp_survived = susp_won = None
        if susp_pid is not None:
            susp_role     = roles.get(susp_pid, Role.VILLAGER).value
            susp_team     = ROLE_TO_TEAM.get(roles.get(susp_pid, Role.VILLAGER))
            susp_survived = record["status"].get(susp_pid) == "ALIVE"
            susp_won      = (susp_team == winner)

        n_susp_survived = sum(1 for p in susp_pids
                              if record["status"].get(p) == "ALIVE")
        n_susp_won      = sum(1 for p in susp_pids
                              if ROLE_TO_TEAM.get(roles.get(p)) == winner)

        # Suspicion calibration from evolution rows
        WOLF_ROLES_SUSP = {"WEREWOLF", "POSSESSED"}
        final_by_player: dict = {}
        for row in all_evo_rows:
            final_by_player[row["target_player"]] = row

        wolf_sigmas = []
        nonwolf_sigmas = []
        wolf_combined = []
        nonwolf_combined = []
        wolf_d    = {i: [] for i in range(1, 6)}
        nonwolf_d = {i: [] for i in range(1, 6)}

        for row in final_by_player.values():
            role_str  = (row.get("target_role") or "").upper()
            is_w      = role_str in WOLF_ROLES_SUSP
            sigma     = float(row.get("sigma", 0.5))
            combined  = float(row.get("combined_score", 0.5))
            if is_w:
                wolf_sigmas.append(sigma)
                wolf_combined.append(combined)
                for i in range(1, 6):
                    wolf_d[i].append(float(row.get(f"d{i}", 0.0)))
            else:
                nonwolf_sigmas.append(sigma)
                nonwolf_combined.append(combined)
                for i in range(1, 6):
                    nonwolf_d[i].append(float(row.get(f"d{i}", 0.0)))

        def _mean(lst):
            return round(sum(lst) / len(lst), 4) if lst else None

        mean_sw = _mean(wolf_sigmas)
        mean_sn = _mean(nonwolf_sigmas)
        sigma_gap = round(mean_sw - mean_sn, 4) \
                    if mean_sw is not None and mean_sn is not None else None

        mean_cw = _mean(wolf_combined)
        mean_cn = _mean(nonwolf_combined)
        combined_gap = round(mean_cw - mean_cn, 4) \
                       if mean_cw is not None and mean_cn is not None else None

        all_final = list(final_by_player.values())
        mean_d = {i: _mean([float(r.get(f"d{i}", 0)) for r in all_final])
                  for i in range(1, 6)}

        # Assemble
        row = {
            "game_id":     record["game_id"],
            "n_players":   record["n_players"],
            "winner":      winner.value,
            "village_win": winner == Team.VILLAGE,
            "n_days":      record["n_days"],
            "n_suspicion_agents":       len(susp_pids),
            "suspicion_agent_id":       f"P{susp_pid}" if susp_pid else "",
            "suspicion_agent_role":     susp_role or "",
            "suspicion_agent_survived": susp_survived,
            "suspicion_agent_won":      susp_won,
            "n_susp_agents_survived":   n_susp_survived,
            "n_susp_agents_won":        n_susp_won,
            "total_votes":           total_votes,
            "correct_wolf_votes":    correct_wolf_votes,
            "false_accusations":     false_accusations,
            "wolf_vote_precision":   wolf_vote_precision,
            "wolf_vote_recall":      wolf_vote_recall,
            "false_accusation_rate": false_accusation_rate,
            "total_wolves":             total_wolves,
            "wolves_executed":          wolves_executed,
            "day_first_wolf_executed":  day_first_wolf_executed,
            "mean_sigma_wolves":        mean_sw,
            "mean_sigma_nonwolves":     mean_sn,
            "sigma_gap":                sigma_gap,
            "mean_combined_wolves":     mean_cw,
            "mean_combined_nonwolves":  mean_cn,
            "combined_gap":             combined_gap,
            "mean_d1": mean_d[1], "mean_d2": mean_d[2], "mean_d3": mean_d[3],
            "mean_d4": mean_d[4], "mean_d5": mean_d[5],
            "mean_d1_wolves":    _mean(wolf_d[1]),
            "mean_d1_nonwolves": _mean(nonwolf_d[1]),
            "mean_d2_wolves":    _mean(wolf_d[2]),
            "mean_d2_nonwolves": _mean(nonwolf_d[2]),
            "mean_d3_wolves":    _mean(wolf_d[3]),
            "mean_d3_nonwolves": _mean(nonwolf_d[3]),
            "mean_d4_wolves":    _mean(wolf_d[4]),
            "mean_d4_nonwolves": _mean(nonwolf_d[4]),
            "mean_d5_wolves":    _mean(wolf_d[5]),
            "mean_d5_nonwolves": _mean(nonwolf_d[5]),
        }

        for atype, short in _TYPE_SHORT.items():
            n = type_total.get(atype, 0)
            c = type_correct.get(atype, 0)
            row[f"{short}_votes"]       = n
            row[f"{short}_correct_pct"] = round(c / n, 4) if n > 0 else None

        return row

    # --- Aggregate summary ------------------------------------------------

    def _print_aggregate(self, rows: list, output_dir: str) -> None:
        n = len(rows)
        if n == 0:
            return

        def _avg(field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else None

        def _fmt(field, decimals=3):
            v = _avg(field)
            return f"{v:.{decimals}f}" if v is not None else "n/a"

        village_wins = sum(1 for r in rows if r.get("village_win") == 1)
        wolf_wins    = n - village_wins
        total_votes  = sum(r.get("total_votes", 0) or 0 for r in rows)
        total_days   = sum(r.get("n_days", 0) or 0 for r in rows)
        has_susp     = any(r.get("n_suspicion_agents", 0) for r in rows)

        print()
        print("=" * 52)
        print(f"  Run complete : {n} games")
        print(f"  Output dir  : {output_dir}")
        print("-" * 52)
        print(f"  Village wins : {village_wins}/{n}  ({village_wins/n*100:.1f}%)")
        print(f"  Wolf wins    : {wolf_wins}/{n}  ({wolf_wins/n*100:.1f}%)")
        print(f"  Avg days     : {total_days/n:.2f}")
        print("-" * 52)
        print(f"  Total votes       : {total_votes}")
        print(f"  Wolf vote prec.   : {_fmt('wolf_vote_precision', 3)}")
        print(f"  Wolf vote recall  : {_fmt('wolf_vote_recall', 3)}")
        print(f"  False accuse rate : {_fmt('false_accusation_rate', 3)}")
        wolves_exec = [r["wolves_executed"] for r in rows if r.get("wolves_executed") is not None]
        avg_we = sum(wolves_exec) / len(wolves_exec) if wolves_exec else None
        print(f"  Avg wolves exec'd : {avg_we:.2f}" if avg_we is not None else "  Avg wolves exec'd : n/a")
        if has_susp:
            print("-" * 52)
            # --- Suspicion agent win rate by role ---
            # Group games by what role the suspicion agent was assigned
            from collections import defaultdict
            susp_role_counts = defaultdict(int)
            susp_role_wins   = defaultdict(int)
            for r in rows:
                role = r.get("suspicion_agent_role")
                won  = r.get("suspicion_agent_won")
                if role is not None and won is not None:
                    susp_role_counts[role] += 1
                    if won:
                        susp_role_wins[role] += 1
            # Non-susp win rate derived from team win rate (role -> team is fixed)
            vw = village_wins / n * 100
            ww = wolf_wins    / n * 100
            nonsusp_roles = [
                ("Villager",  vw),
                ("Seer",      vw),
                ("Werewolf",  ww),
                ("Possessed", ww),
            ]
            print("  Win rate by agent type + role:")
            role_order = ["VILLAGER", "SEER", "POSSESSED"]
            for role in role_order:
                cnt = susp_role_counts.get(role, 0)
                if cnt == 0:
                    continue
                wins = susp_role_wins.get(role, 0)
                pct  = wins / cnt * 100
                label = role.capitalize()
                print(f"    Susp     | {label:<10}: {pct:.1f}%  (n={cnt})")
            print(f"    ----------")
            for label, pct in nonsusp_roles:
                print(f"    Non-susp | {label:<10}: {pct:.1f}%")
            print("-" * 52)
            print(f"  Sigma gap          : {_fmt('sigma_gap', 4)}")
            print(f"  Combined score gap : {_fmt('combined_gap', 4)}")
            susp_won      = sum(r.get("n_susp_agents_won", 0) or 0 for r in rows)
            susp_total    = sum(r.get("n_suspicion_agents", 0) or 0 for r in rows)
            total_winners = sum(5 if r.get("village_win") == 1 else 3 for r in rows)
            nonsusp_won   = total_winners - susp_won
            nonsusp_total = sum(
                (r.get("n_players", 8) - (r.get("n_suspicion_agents", 0) or 0))
                for r in rows
            )
            if susp_total > 0:
                print(f"  Susp-agent wins    : {susp_won}/{susp_total}  ({susp_won/susp_total*100:.1f}%)")
            if nonsusp_total > 0:
                print(f"  Non-susp wins      : {nonsusp_won}/{nonsusp_total}  ({nonsusp_won/nonsusp_total*100:.1f}%)")
        print("=" * 52)
        print()

    # --- Output helpers ---------------------------------------------------

    def _write_summary(self, rows: list, output_dir: str) -> None:
        path = os.path.join(output_dir, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_config_echo(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "config_used.json")
        serialisable = {k: v for k, v in self._config.items()
                        if isinstance(v, (int, float, str, bool, list,
                                         dict, type(None)))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

    def _write_game_summary(self, record, roles, susp_pid, game_dir):
        summary = {
            "game_id":              record["game_id"],
            "n_players":            record["n_players"],
            "winner":               record["winner"].value,
            "n_days":               record["n_days"],
            "roles":                record["roles"],
            "agents":               record["agent_names"],
            "final_status":         record["status"],
            "suspicion_agent_id":   susp_pid,
            "suspicion_agent_role": record.get("suspicion_agent_role"),
        }
        path = os.path.join(game_dir, "game_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
