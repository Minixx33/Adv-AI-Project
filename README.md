# Werewolf AI — Suspicion-Aware Agent Research

A self-contained Python simulation engine for the social-deduction game *Werewolf*, built for MLR 555 (Advanced AI). The project investigates whether a **suspicion module** — a pluggable behavioral analysis layer — improves a village agent's ability to identify and eliminate werewolves.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Quick Start](#4-quick-start)
5. [Game Rules & Roles](#5-game-rules--roles)
6. [Agent Types](#6-agent-types)
7. [The Suspicion Module](#7-the-suspicion-module)
8. [Configuration Reference](#8-configuration-reference)
9. [CLI Reference](#9-cli-reference)
10. [Running Experiments](#10-running-experiments)
11. [Ablation Studies](#11-ablation-studies)
12. [Statistical Analysis](#12-statistical-analysis)
13. [Output Files](#13-output-files)
14. [Talk Protocol](#14-talk-protocol)
15. [Grid Search (Hyperparameter Optimization)](#15-grid-search-hyperparameter-optimization)
16. [Legacy AIWolf Server Mode](#16-legacy-aiwolf-server-mode)
17. [Project Architecture Deep Dive](#17-project-architecture-deep-dive)

---

## 1. Project Overview

### What is Werewolf?

Werewolf (also known as Mafia) is a social-deduction game where **villagers** try to identify and eliminate hidden **werewolves** through public discussion and voting, while werewolves secretly eliminate villagers each night. Deception, argument analysis, and coalition inference are all required to win.

### Research Question

> *Does a behavioral suspicion module — one that tracks linguistic and voting patterns across game rounds — improve a village agent's wolf-identification accuracy compared to belief-only baselines?*

### The Suspicion Module

The suspicion module is a **modular, pluggable add-on** that runs alongside any compatible base agent (Heuristic, Bayesian, or MCTS). It maintains a suspicion score σ(p) ∈ [0, 1] for every opponent, updated by five detectors that monitor behavioral patterns in the talk and vote logs. The combined decision score blends the base agent's belief with the suspicion score:

```
score(p) = w1 * base_belief(p) + w2 * σ(p)
```

### Key Design Decisions

- **No Java server required.** Everything runs in-process in pure Python — fast, fully observable, reproducible.
- **Mirrors the AIWolf interface.** Agent methods (`initialize`, `update`, `day_start`, `talk`, `vote`, `divine`, `attack`, `finish`) are compatible with the original AIWolf platform for future interoperability.
- **Modular suspicion.** The module is not baked into any agent — any compatible agent can be upgraded with a single config flag.
- **Suspicion agent is never a werewolf.** This is a hard constraint to keep the research question clean: the module is designed to detect deception, not to practice it.

---

## 2. Repository Structure

```
Adv-AI-Project/
├── README.md
├── requirements.txt
│
├── Model_V2/                          # Main simulation engine
│   ├── run_game.py                    # CLI entry point
│   ├── stats_analysis.py              # Post-hoc statistical tests
│   │
│   ├── werewolf/                      # Core game engine
│   │   ├── game.py                    # WerewolfGame class (game loop)
│   │   ├── gameinfo.py                # GameInfo, Talk, Vote, Judge structs
│   │   ├── const.py                   # Role, Status, Species enums
│   │   ├── role_assigner.py           # Role assignment with suspicion constraint
│   │   ├── agent.py                   # AbstractAgent base class
│   │   └── agents/
│   │       ├── random_agent.py        # Floor baseline (random votes)
│   │       ├── heuristic_agent.py     # Accusation-count heuristic
│   │       ├── bayesian_agent.py      # Bayesian belief updater
│   │       ├── mcts_agent.py          # Monte Carlo Tree Search
│   │       └── logic_agent.py         # Symbolic forward-chaining
│   │
│   ├── suspicion/                     # Suspicion module
│   │   ├── module.py                  # SuspicionModule class (σ update logic)
│   │   ├── detectors.py               # D1–D5 detector implementations
│   │   ├── enhanced_agent.py          # Wraps a base agent + module
│   │   └── online_learner.py          # Adaptive weight learning
│   │
│   ├── runner/                        # Experiment infrastructure
│   │   ├── game_runner.py             # Runs N games, manages composition
│   │   ├── grid_search.py             # Two-stage hyperparameter search
│   │   ├── conversation_logger.py     # Full talk log per game
│   │   ├── suspicion_tracer.py        # Per-round reasoning trace
│   │   ├── evolution_logger.py        # σ time-series CSV
│   │   └── xlsx_exporter.py           # Excel summary export
│   │
│   ├── ablations/                     # Ablation experiment scripts
│   │   ├── run_ablation_detector.py   # Leave-one-out detector ablation
│   │   ├── run_ablation_weights.py    # w1/w2 decision-weight sweep
│   │   ├── run_ablation_adversarial.py# Wolf strategy ablation (RQ3)
│   │   └── run_ablation_learner.py    # Online learner vs fixed weights
│   │
│   ├── configs/
│   │   └── default.json               # Default game configuration
│   │
│   ├── results/                       # Auto-created output directory
│   │
│   └── tests/
│       ├── test_game.py
│       ├── test_agents.py
│       └── test_suspicion.py
│
├── configs/                           # Named experiment configurations
│   ├── bayesian_baseline.json
│   ├── bayesian_fullsus.json
│   ├── bayesian_2sus.json
│   ├── bayesian_4sus.json
│   ├── bayesian_6sus.json
│   ├── heuristic_baseline.json
│   ├── heuristic_fullsus.json
│   ├── heuristic_2sus.json
│   ├── heuristic_4sus.json
│   ├── heuristic_6sus.json
│   ├── mcts_baseline.json
│   ├── mcts_fullsus.json
│   ├── mcts_2sus.json
│   ├── mcts_4sus.json
│   ├── mcts_6sus.json
│   ├── logic_baseline.json
│   └── random_baseline.json
│
├── run_baseline_experiments.sh        # Run all baseline conditions (WSL)
├── run_fullsus_experiments.sh         # Run all full-suspicion conditions
├── run_mixed2_experiments.sh          # Run 2-suspicion-agent conditions
├── run_4sus_experiments.sh            # Run 4-suspicion-agent conditions
├── run_6sus_experiments.sh            # Run 6-suspicion-agent conditions
└── run_ablations.sh                   # Run all 4 ablation studies
```

---

## 3. Prerequisites & Installation

### Requirements

- **Python 3.8+**
- **pip** packages:

```bash
pip install scipy numpy pandas openpyxl
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

> **No Java or external server needed.** The entire simulation runs in Python.

### Cloning the repository

```bash
git clone <repo-url>
cd Adv-AI-Project
```

---

## 4. Quick Start

### Run a single game (default config)

```bash
cd Model_V2
python run_game.py --config configs/default.json
```

### Run 100 games and save output

```bash
python Model_V2/run_game.py \
    --config Model_V2/configs/default.json \
    --n-games 100 \
    --seed 42 \
    --output results/my_first_run/
```

### Run with verbose output (see every vote and talk token)

```bash
python Model_V2/run_game.py \
    --config Model_V2/configs/default.json \
    --n-games 5 \
    --verbose
```

### Baseline: no suspicion agent

```bash
python Model_V2/run_game.py \
    --config configs/bayesian_baseline.json \
    --n-games 100 \
    --seed 42
```

---

## 5. Game Rules & Roles

The simulation supports 5-, 8-, and 10-player configurations. Each game alternates between **day phases** (public discussion + voting) and **night phases** (secret wolf attack + seer divine).

### Role Distributions

| Players | Villagers | Werewolves | Seer | Possessed | Bodyguard |
|---------|-----------|------------|------|-----------|-----------|
| 5       | 2         | 1          | 1    | 1         | —         |
| **8** *(default)* | **4** | **2** | **1** | **1** | — |
| 10      | 5         | 2          | 1    | 1         | 1         |

### Roles Explained

**Villager** — No special ability. Must deduce wolves through argument analysis and voting patterns.

**Werewolf** — Knows who their fellow wolves are. Each night, wolves collectively choose one player to eliminate. During the day they act as villagers to avoid detection. Wolf talk strategies include:
- *Bus-Driver* (prob. 0.2): Occasionally accuses a fellow wolf to build credibility.
- *False-Claimer* (prob. 0.5): Claims to be Seer on Day 1 and fabricates divine results.

**Seer** — Each night, privately learns whether one chosen player is Human or Werewolf. Should strategically reveal (or conceal) this information.

**Possessed** — On the wolf team but appears Human to the Seer. Acts as a decoy and disinformation agent.

**Bodyguard** *(10-player only)* — Each night, protects one player from wolf attack. Cannot protect themselves or the same player two nights in a row.

### Win Conditions

- **Village wins** when all werewolves are eliminated.
- **Wolf team wins** when werewolves equal or outnumber remaining villagers.

---

## 6. Agent Types

All agents implement the same interface: `initialize`, `update`, `day_start`, `talk`, `vote`, `whisper`, `attack`, `divine`, `finish`.

### Random Agent
The floor baseline. Votes uniformly at random. Says "Skip" during talk. No belief tracking. Cannot be enhanced with the suspicion module.

### Heuristic Agent
Tracks accusation counts per player and votes for the most-accused alive player. During talk, it declares VOTE, AGREEs when someone attacks its target, and issues ATTACK tokens on the most-accused. **Compatible with suspicion module.**

### Bayesian Agent *(recommended primary baseline)*
Maintains P(role | observations) for every opponent. Updates beliefs via Bayes' rule with Monte Carlo likelihood estimation. Votes for the player with the highest P(wolf). Seer divine results update beliefs with certainty. During talk: emits ESTIMATE, ATTACK, BECAUSE, and VOTE targeting its top wolf candidate. **Compatible with suspicion module.**

### MCTS Agent
Uses Monte Carlo Tree Search with UCT (Upper Confidence Trees) to evaluate vote decisions. Runs up to 500 simulations per decision (configurable), with a tree depth of 5. UCB1 balances exploration vs. exploitation. During talk: targets its MCTS-preferred candidate with ESTIMATE and VOTE, and AGREE/DISAGREEs based on whether dominant accusations match its tree search. **Compatible with suspicion module.**

### Logic-Based Agent
Maintains a knowledge base of observed facts and uses forward-chaining inference. Claims are stored as `Claims(P3, Wolf(P5))` rather than direct facts. Handles contradictions — if two players claim the same exclusive role, derives that at least one is on the wolf team. **Cannot be enhanced with suspicion** (deals in certainties, not degrees).

### Suspicion-Enhanced Agent (`enhanced_agent.py`)
A wrapper that layers the suspicion module on top of any compatible base agent (Heuristic, Bayesian, or MCTS). The base agent provides its belief score; the suspicion module provides σ(p); the wrapper combines them as `score(p) = w1 * belief(p) + w2 * σ(p)`.

---

## 7. The Suspicion Module

### State

Each opponent `p` has a suspicion score `σ(p) ∈ [0, 1]`, initialized to 0.5 (neutral). Higher values indicate more suspicious behavior.

### Five Detectors

The module computes a composite evidence value `E(p)` from five behavioral detectors:

**D1 — Vote-Accusation Mismatch**
Detects players who publicly attack one target but vote for a different one — a classic wolf deflection tactic.
```
D1(p) = count(ATTACK by p targeting X where p.vote ≠ X) / total_attacks_by(p)
```

**D2 — Bandwagon Index**
Detects players who join existing majorities rather than forming independent judgments. Wolves sometimes follow crowds to avoid standing out.
```
D2(p) = count(AGREE tokens by p joining an already-dominant accusation) / total_agree_by(p)
```

**D3 — Pressure-Triggered Accusations**
Detects players who become aggressive *specifically when they themselves are under scrutiny* — a defense mechanism commonly used by wolves.
```
D3(p) = Pearson correlation between [pressure_on_p per round] and [accusations_by_p per round]
```

**D4 — Protection Patterns**
Detects players who consistently shield the same other player using DEFEND, DISAGREE-on-attacks, or NOT tokens. Wolves often protect each other.
```
D4(p) = max over q of (defends(p, q) / total_rounds)
```

**D5 — Claim Consistency**
Detects players whose COMINGOUT or DIVINED claims are contradicted by game outcomes. This is the strongest signal when a False-Claimer wolf is present.
```
D5(p) = contradicted_claims(p) / total_claims_by(p)
```

### Composite Evidence

```
E(p) = a1*D1(p) + a2*D2(p) + a3*max(D3(p), 0) + a4*D4(p) + a5*D5(p)
```

Default weights: `a1=0.30, a2=0.15, a3=0.20, a4=0.20, a5=0.15`

### Asymmetric Update Rule

Trust is slow to rebuild but fast to erode:

```
If E(p) > 0.5:  σ(p) ← min(1.0, σ(p) + 0.15 * E(p))       # suspicious → fast increase
If E(p) ≤ 0.5:  σ(p) ← max(0.0, σ(p) - 0.05 * (1 - E(p))) # innocent → slow decrease
```

### Combined Decision Score

```
score(p) = w1 * base_belief(p) + w2 * σ(p)
```

Default: `w1=0.6, w2=0.4`. The suspicion agent votes for `argmax score(p)`.

### Hard Constraint: Suspicion Agent Cannot Be Werewolf

The suspicion-enhanced agent is **never assigned the Werewolf role**. Role assignment works as follows: (1) the suspicion agent's slot is identified; (2) wolf roles are assigned to non-suspicion slots only; (3) remaining roles fill the remaining slots. The suspicion agent *can* be Possessed (wolf team, but not literally wolf), which tests whether the module correctly handles its own deceptive position.

---

## 8. Configuration Reference

Games are configured via JSON files. The full set of options:

```json
{
  "games": 10000,
  "players": 8,
  "suspicion_agent_type": "bayesian",
  "agent_composition": {
    "random": 0,
    "heuristic": 0,
    "bayesian": 7,
    "mcts": 0,
    "bayesian_with_suspicion": 1
  },
  "suspicion_weights": {
    "a1": 0.30,
    "a2": 0.15,
    "a3": 0.20,
    "a4": 0.20,
    "a5": 0.15,
    "w1": 0.6,
    "w2": 0.4
  },
  "mcts_simulations": 500,
  "mcts_depth": 5,
  "output_dir": "./results",
  "max_talk_rounds": 5,
  "seed": null,
  "verbose": false,
  "wolf_strategies": {
    "bus_driver_probability": 0.2,
    "false_claimer_probability": 0.5
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `games` | int | Number of games to simulate |
| `players` | int | Must be 5, 8, or 10 |
| `suspicion_agent_type` | string | Base agent for the suspicion-enhanced slot: `"bayesian"`, `"heuristic"`, `"mcts"`, or `"none"` |
| `agent_composition` | object | Map of agent type → count. Must sum to `players` |
| `suspicion_weights` | object | Detector weights a1–a5 and decision weights w1, w2 |
| `mcts_simulations` | int | Simulations per MCTS decision |
| `mcts_depth` | int | Tree search depth for MCTS |
| `output_dir` | string | Where to write results |
| `max_talk_rounds` | int | Talk rounds per day phase |
| `seed` | int or null | Random seed (null = random per run) |
| `verbose` | bool | Print detailed per-game progress |
| `wolf_strategies.bus_driver_probability` | float | Probability a wolf uses the bus-driver tactic this game |
| `wolf_strategies.false_claimer_probability` | float | Probability a wolf uses the false-claimer tactic |

### Pre-built Configs (`configs/`)

| Config | Description |
|--------|-------------|
| `bayesian_baseline.json` | 8 Bayesian agents, no suspicion |
| `bayesian_fullsus.json` | All 8 agents use suspicion (upper bound) |
| `bayesian_2sus.json` | 6 Bayesian + 2 Bayesian+Suspicion |
| `bayesian_4sus.json` | 4 Bayesian + 4 Bayesian+Suspicion |
| `bayesian_6sus.json` | 2 Bayesian + 6 Bayesian+Suspicion |
| `heuristic_*` | Same conditions for Heuristic base agent |
| `mcts_*` | Same conditions for MCTS base agent |
| `logic_baseline.json` | 8 Logic agents, no suspicion |
| `random_baseline.json` | 8 Random agents (floor check) |

---

## 9. CLI Reference

```
python Model_V2/run_game.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config PATH` | str | required | Path to JSON config file |
| `--n-games N` | int | from config | Override number of games |
| `--players N` | int | from config | Override player count (must be 5, 8, or 10) |
| `--suspicion-agent TYPE` | str | from config | Override suspicion agent type |
| `--seed N` | int | from config | Random seed for reproducibility |
| `--output DIR` | str | from config | Output directory |
| `--run-name NAME` | str | auto | Label for this run (used in output path) |
| `--verbose` | flag | false | Print game-by-game progress |
| `--grid-search` | flag | false | Run hyperparameter grid search instead |

### Examples

```bash
# Default 100-game run
python Model_V2/run_game.py --config Model_V2/configs/default.json

# 10,000-game reproducible run
python Model_V2/run_game.py \
    --config configs/bayesian_baseline.json \
    --seed 42 \
    --n-games 10000 \
    --output results/bayesian_baseline/seed_42/

# No suspicion (baseline)
python Model_V2/run_game.py \
    --config configs/bayesian_baseline.json \
    --n-games 1000

# Full suspicion (upper bound)
python Model_V2/run_game.py \
    --config configs/bayesian_fullsus.json \
    --n-games 1000

# Verbose single-game debug
python Model_V2/run_game.py \
    --config Model_V2/configs/default.json \
    --n-games 1 \
    --verbose
```

---

## 10. Running Experiments

The paper's main experiment compares five conditions across three seeds (42, 123, 456), with 10,000 games each:

| Condition | Description | Config pattern |
|-----------|-------------|----------------|
| **Baseline** | No suspicion agents | `*_baseline.json` |
| **Full Suspicion** | All compatible agents use suspicion | `*_fullsus.json` |
| **Mixed-2** | 2 suspicion-enhanced agents | `*_2sus.json` |
| **Mixed-4** | 4 suspicion-enhanced agents | `*_4sus.json` |
| **Mixed-6** | 6 suspicion-enhanced agents | `*_6sus.json` |

### Running from WSL (recommended)

The shell scripts handle all conditions and seeds automatically. Run them from the project root:

```bash
# Baseline (all agent types, 3 seeds each)
bash run_baseline_experiments.sh

# Full suspicion
bash run_fullsus_experiments.sh

# Mixed-population dose-response
bash run_mixed2_experiments.sh
bash run_4sus_experiments.sh
bash run_6sus_experiments.sh
```

> **Note:** The shell scripts use paths relative to WSL mounts (`/mnt/c/Users/...`). If you're running on macOS or native Linux, update the `PROJECT_DIR` variable at the top of each script.

### Running a single condition manually

```bash
python Model_V2/run_game.py \
    --config configs/bayesian_baseline.json \
    --seed 42 \
    --run-name "bayesian_baseline_seed42" \
    --n-games 10000 \
    --players 8
```

Repeat with `--seed 123` and `--seed 456` for the other seeds.

---

## 11. Ablation Studies

Four ablation experiments probe *why* the suspicion module works (or doesn't):

### Ablation 1 — Detector Leave-One-Out

Runs 6 conditions (all detectors active + one condition with each detector zeroed out in turn) to measure each detector's marginal contribution to village win rate.

### Ablation 2 — Decision-Weight Sweep (w1/w2)

Sweeps w1 from 0.0 to 1.0 in steps of 0.2 (w2 = 1 − w1) to find the optimal blend between base belief and suspicion score.

### Ablation 3 — Adversarial Wolf Strategies (RQ3)

Tests the suspicion module against three wolf behavior profiles (passive, bus-driver, false-claimer) with suspicion on and off — revealing which wolf behaviors the module counters most effectively.

### Ablation 4 — Online Learner vs. Fixed Weights

Compares static a1–a5 weights against an online-learning agent that adapts its detector weights based on whether its predictions proved correct.

### Running all ablations

```bash
bash run_ablations.sh
```

Options:

```bash
# Custom output directory
bash run_ablations.sh --output /path/to/output

# Dry run — print commands without executing
bash run_ablations.sh --dry-run
```

Results land in `Model_V2/results/ablations/`. A timestamped master log is written to `ablations_master.log`.

**Game counts per ablation:**
- Ablations 1, 2, 3: 10,000 games × 3 seeds × number of conditions
- Ablation 4 (online learner): 2,000 games × 3 seeds (sequential; capped to keep runtime manageable)

---

## 12. Statistical Analysis

After all experiment runs are complete, pool results across seeds and run significance tests:

```bash
python Model_V2/stats_analysis.py \
    --baseline  results/baseline/ \
    --full      results/full_suspicion/ \
    --mixed-2   results/mixed_2/ \
    --mixed-4   results/mixed_4/ \
    --mixed-6   results/mixed_6/ \
    --output    stats_results.csv
```

The script recursively finds all `summary.csv` files under each supplied directory (any seed subdirectory structure works), pools all games, and runs:

- **Two-proportion z-test** on village win rate (effect size: Cohen's h)
- **Mann-Whitney U test** on continuous metrics: σ gap, wolf vote precision/recall, false accusation rate, per-detector means D1–D5 (effect size: rank-biserial r)

Comparisons run automatically: baseline vs. all conditions, plus the mixed-population dose-response chain. Significance stars (`***` p < 0.001 / `**` p < 0.01 / `*` p < 0.05 / `ns`) and a seed-level stability table are printed to stdout.

```bash
# Change significance threshold
python Model_V2/stats_analysis.py ... --alpha 0.01
```

---

## 13. Output Files

Each run produces the following in its output directory:

| File | Contents |
|------|----------|
| `summary.csv` | Aggregate metrics across all games — main input to `stats_analysis.py` |
| `config_used.json` | Echo of the configuration used for this run |
| `conversation.txt` | Full talk log for every game (all tokens, all rounds) |
| `suspicion_trace.txt` | Per-round σ scores and detector values with token evidence (suspicion games only) |
| `suspicion_evolution.csv` | Time-series suspicion data for plotting (one row per player per round) |
| `game_summary.json` | Per-game structured results |
| `*.xlsx` | Excel summary exported by XlsxExporter |

### `summary.csv` columns

```
game_id, n_players, winner, n_days, suspicion_agent_id, suspicion_agent_role,
suspicion_agent_survived, suspicion_agent_won,
correct_wolf_votes, total_votes, false_accusations
```

### `suspicion_evolution.csv` columns

```
day, round, target_player, target_role, sigma, trust, belief_wolf,
combined_score, d1, d2, d3, d4, d5, E, delta_sigma
```

Where `trust = 1 - σ`. This enables time-series plots of how suspicion evolves and which detectors fire when.

### Conversation log format

```
=== GAME 42 — 8 PLAYERS ===
Roles (revealed at end):
  P1: Villager (HeuristicAgent)
  P2: Werewolf (BayesianAgent)
  P3: Villager (BayesianAgent+Suspicion)
  ...

--- DAY 1 ---
[Round 1]
P6: COMINGOUT P6 SEER
P6: DIVINED P5 WEREWOLF
P6: BECAUSE divine_result
P6: VOTE P5
P6: OVER

P5: DISAGREE P6
P5: COMINGOUT P5 SEER
P5: DIVINED P3 WEREWOLF
...

[Voting Phase — Day 1]
P1 → P5   P2 → P3   ...
Tally: P5(5), P3(2), P6(1)
Eliminated: P5 (was Werewolf)

=== RESULT: VILLAGE WINS in 4 days ===
```

---

## 14. Talk Protocol

Each agent gets up to **5 talk tokens per round**, cycling until it emits `OVER`, uses all 5 tokens, or emits `SKIP`. An 8-player game produces ~30 tokens per round.

### Token Reference

| Token | Meaning | Example |
|-------|---------|---------|
| `VOTE Pn` | Declares intention to vote for Pn | `VOTE P5` |
| `ESTIMATE Pn ROLE` | Claims Pn is a certain role | `ESTIMATE P3 WEREWOLF` |
| `COMINGOUT Pn ROLE` | Reveals own role | `COMINGOUT P6 SEER` |
| `DIVINED Pn RESULT` | Reports Seer divine result | `DIVINED P5 WEREWOLF` |
| `ATTACK Pn` | Strong public accusation | `ATTACK P5` |
| `DEFEND Pn` | Defends Pn against accusations | `DEFEND P3` |
| `AGREE Pn` | Agrees with Pn's last statement | `AGREE P3` |
| `DISAGREE Pn` | Disagrees with Pn's last statement | `DISAGREE P5` |
| `REQUEST Pn action` | Asks Pn to take an action | `REQUEST P3 reveal_role` |
| `INQUIRE Pn` | Questions Pn's behavior or claim | `INQUIRE P5` |
| `BECAUSE reason` | Justification for previous statement | `BECAUSE seer_claim_conflict` |
| `NOT Pn ROLE` | Asserts Pn is NOT a particular role | `NOT P3 SEER` |
| `IDENTIFIED Pn ROLE` | Medium identifies a dead player's role | `IDENTIFIED P5 WEREWOLF` |
| `GUARDED Pn` | Bodyguard claims to have protected Pn | `GUARDED P3` |
| `SKIP` | Declines to speak this turn | — |
| `OVER` | Done talking this round | — |

### BECAUSE Reason Codes

`divine_result` · `seer_claim_conflict` · `vote_pattern` · `bandwagon` · `defended_wolf` · `claim_contradiction` · `silent_player` · `aggressive_accuser` · `protect_seer` · `last_resort`

---

## 15. Grid Search (Hyperparameter Optimization)

The two-stage grid search finds optimal detector and decision weights.

**Stage 1 — Detector weights (a1–a5):** Latin Hypercube Sampling over ~200 configurations. Each weight varies 0.0–0.5; weights are normalized to sum to 1.0. 2,000 games per configuration.

**Stage 2 — Decision weights (w1, w2):** Using the best Stage 1 weights, sweeps w1 from 0.0 to 1.0 in steps of 0.1 (w2 = 1 − w1). 11 configurations × 2,000 games each.

```bash
python -c "
from Model_V2.runner.grid_search import GridSearch
import json
config = json.load(open('Model_V2/configs/default.json'))
gs = GridSearch(config, n_games=2000, n_samples=200, seed=0)
gs.run('results/grid_search/')
"
```

Or via CLI:

```bash
python Model_V2/run_game.py \
    --config Model_V2/configs/default.json \
    --grid-search \
    --output results/grid_search/
```

Results are saved to `results/grid_search/grid_search_results.csv`. The top 5 configurations by village win rate are printed at completion.

---

## 16. Legacy AIWolf Server Mode

The original AIWolfPy interface is still available for connecting Python agents to the official Java-based AIWolf competition server.

### Requirements

- JDK 11
- AIWolf platform (download from [aiwolf.org/server](http://www.aiwolf.org/server/))

### Steps

1. Start the server: `./StartServer.sh`
2. In another terminal, start the GUI client: `./StartGUIClient.sh`
   - Select `sampleclient.jar` and connect agents for each desired player slot.
3. Run the Python agent: `./python_sample.py -h [hostname] -p [port]`
4. Press "Start Game" on the server application.

Logs are saved to `./log/` and can be visualized with the AIWolf log viewer.

For the competition server, make sure your client name matches your account name. Available Python packages are listed at [aiwolf.org/python_modules](http://aiwolf.org/python_modules). The competition rules forbid multiple threads — if using TensorFlow, ensure your code is single-threaded (see [this post](http://aiwolf.org/archives/1951)).

---

## 17. Project Architecture Deep Dive

### Game Loop (`werewolf/game.py`)

```
initialize all agents
while game_not_over:
    day_start()
    for round in 1..max_talk_rounds:
        for each alive agent (randomized order):
            while agent has budget (≤5 tokens) and hasn't sent OVER/SKIP:
                token = agent.talk()
                broadcast token to all agents via update()
    voting = [agent.vote() for each alive agent]
    eliminate player with most votes (ties broken randomly)
    if game_over: break

    night:
        seer.divine()
        wolves.attack()   ← whisper phase
        bodyguard.guard() ← 10-player only
        apply night results
        update all agents

finish(all agents)
```

### Belief Update (Bayesian Agent)

On each observation, the Bayesian agent updates P(role | obs) for every opponent:

```
P(p = wolf | obs) ∝ P(obs | p = wolf) * P(p = wolf)
```

Likelihood `P(obs | role)` is estimated via Monte Carlo sampling over possible world states consistent with the observation.

### MCTS Decision Making

For each vote decision, MCTS: (1) samples a set of possible role assignments consistent with observations; (2) simulates games forward using simplified heuristic policies; (3) backs up win rates to the root; (4) returns `argmax expected_village_win_rate` as the vote target.

### Performance Targets

| Scenario | Target |
|----------|--------|
| Single 8-player game | < 10 seconds |
| 10,000 games (8-player) | < 12 hours on a standard laptop |
| Grid search Stage 1 (200 configs × 2,000 games) | Overnight |
| Peak memory | < 500 MB |

---

## Changelog

### Model_V2 (current)
- Self-contained Python simulation (no Java server required)
- Five agent types: Random, Heuristic, Bayesian, MCTS, Logic-Based
- Suspicion module with five behavioral detectors (D1–D5)
- Expanded talk protocol (16 token types, up to 5 tokens per agent per round)
- Wolf talk strategies: Bus-Driver and False-Claimer
- Full logging: conversation log, suspicion trace, suspicion evolution CSV, XLSX export
- Two-stage grid search for hyperparameter optimization
- Online learner ablation
- Statistical analysis pipeline (z-test + Mann-Whitney)

### AIWolfPy v0.4.9a
- Added English support material

### AIWolfPy v0.4.9
- Changed differential structure (diff_data) into a DataFrame

### AIWolfPy v0.4.4
- Removed daily_finish
- Added update callback (with request parameter)
- Connecting is now done through an instance, not a class

### AIWolfPy v0.4.0
- Python 3 support
- Simplified file structure

---

*This project was forked from [AIWolfPy](https://github.com/k-harada) by Kei Harada. The Model_V2 simulation engine and suspicion module are original research work for MLR 555 (Advanced AI).*
