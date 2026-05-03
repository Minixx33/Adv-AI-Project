# Suspicion-Aware Werewolf AI — Architecture v2

## Context for Claude Code

This document is the build specification for the **second version** of a self-contained Python Werewolf game engine built for MLR 555 (Advanced AI). It supersedes any earlier architecture document. Claude Code should use this as the authoritative reference. The project has no Java server, no socket connections — everything runs in-process in Python.

The game engine still mirrors the AIWolf platform's agent interface (same method names, same GameInfo structure) for compatibility, but executes locally for speed and full observability.

---

## 1. Five Base Agents

All agents implement the same interface (mirroring AIWolf): `initialize`, `update`, `day_start`, `talk`, `vote`, `whisper`, `attack`, `divine`, `finish`.

### Random Agent (Floor Baseline)
- Votes uniformly at random among alive players
- Says "Skip" during talk phase
- No belief tracking
- Cannot be enhanced with suspicion

### Heuristic Agent (Simple Baseline)
- Tracks accusation counts per player
- Votes for the most-accused alive player
- Talk: declares VOTE for current most-accused
- **Compatible with suspicion module**

### Bayesian Agent (Strong Baseline)
- Maintains P(role | observations) for every player
- Updates beliefs via Bayes' rule using Monte Carlo likelihood estimation
- Votes for argmax P(p = wolf)
- Seer divine results update beliefs with certainty
- **Compatible with suspicion module**

### MCTS Agent (State-of-the-Art Baseline)
- Uses Monte Carlo Tree Search with UCT (Upper Confidence Trees)
- For each decision, runs N simulations of future game trajectories
- UCB1 formula balances exploration and exploitation
- Limit: 500 simulations per decision (configurable)
- Limit: depth 5 in the tree (configurable)
- **Compatible with suspicion module**

### Logic-Based Agent (Symbolic Reasoning)
- Maintains a knowledge base of observed facts
- Forward chaining inference with tagged hypothetical reasoning
- Claims stored as `Claims(P3, Wolf(P5))` — never as direct facts
- Contradiction handling: if two players claim same exclusive role, derives "at least one is on wolf team"
- Cannot be enhanced with suspicion (deals in certainties, not degrees)

---

## 2. The Suspicion Module (Modular Add-On)

**Critical requirement:** The suspicion module is implemented as a **separate, pluggable component**. Any compatible base agent (Heuristic, Bayesian, MCTS) can optionally use it.

State per player: `σ(p) ∈ [0, 1]` — initialized to 0.5

### Five Detectors

**D1 — Vote-Accusation Mismatch:**
`D1(p) = mismatches(p) / rounds(p)`
Tracks: did player accuse X but vote for Y ≠ X?

**D2 — Bandwagon Index:**
`D2(p) = times_followed_majority(p) / applicable_rounds(p)`
Tracks: how often does player join the existing majority?

**D3 — Pressure-Triggered Accusations:**
`D3(p) = correlation(suspicion_on_p, accusations_by_p)`
Tracks: does player accuse only when under pressure?

**D4 — Protection Patterns:**
`D4(p) = max over q of (times_p_defended_q / rounds)`
Tracks: does player consistently shield a specific other?

**D5 — Claim Consistency:**
`D5(p) = contradicted_claims(p) / total_claims(p)`
Tracks: are role claims contradicted by game outcomes?

### Composite Evidence
```
E(p) = a1*D1(p) + a2*D2(p) + a3*max(D3(p),0) + a4*D4(p) + a5*D5(p)
```

### Asymmetric Update (trust slow to rebuild)
```
If E(p) > 0.5:  σ(p) ← min(1.0, σ(p) + 0.15 * E(p))
If E(p) ≤ 0.5:  σ(p) ← max(0.0, σ(p) - 0.05 * (1 - E(p)))
```

### Combined Decision
```
score(p) = w1 * base_belief(p) + w2 * σ(p)
```
Vote target: `argmax score(p)`

**All weights (a1-a5, w1, w2) must be configurable, not hardcoded** — they will be optimized via grid search.

---

## 3. NEW REQUIREMENTS (v2)

### 3.1 PowerShell CLI — Configurable Game Parameters

Build a CLI entry point `run_game.py` that accepts the following arguments via PowerShell:

```powershell
python run_game.py --games 100 --players 8 --suspicion-agent bayesian --output ./results
```

**Required arguments:**

| Argument | Type | Options | Default |
|----------|------|---------|---------|
| `--games` | int | Any positive integer | 100 |
| `--players` | int | 5, 8, or 10 only | 8 |
| `--suspicion-agent` | str | `heuristic`, `bayesian`, `mcts`, `none` | `bayesian` |
| `--config` | str | Path to JSON config file (overrides other args) | None |
| `--grid-search` | flag | If set, runs grid search instead of normal games | False |
| `--output` | str | Output directory for logs | `./results` |
| `--seed` | int | Random seed for reproducibility | None |
| `--verbose` | flag | Print detailed game progress to console | False |

**Validation:** If `--players` is not in {5, 8, 10}, exit with clear error message. If `--suspicion-agent` is `none`, run baseline games without any suspicion-enhanced agent.

### 3.2 Player Count Configurations

Three role distributions must be supported:

**5-player setup:**
- 2 Villagers, 1 Werewolf, 1 Seer, 1 Possessed
- Single wolf — D4 (Protection Patterns) detector inactive

**8-player setup (recommended primary):**
- 4 Villagers, 2 Werewolves, 1 Seer, 1 Possessed
- Multi-wolf coordination possible

**10-player setup:**
- 5 Villagers, 2 Werewolves, 1 Seer, 1 Possessed, 1 Bodyguard
- Bodyguard can protect one player from wolf attacks each night
- More complex inference, longer games

### 3.3 Agent Composition Configuration

Beyond the CLI arguments, support a JSON config file specifying exact agent compositions:

```json
{
  "games": 1000,
  "players": 8,
  "suspicion_agent_type": "bayesian",
  "agent_composition": {
    "random": 1,
    "heuristic": 2,
    "bayesian": 1,
    "mcts": 2,
    "logic_based": 1,
    "bayesian_with_suspicion": 1
  },
  "suspicion_weights": {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
    "w1": 0.6, "w2": 0.4
  },
  "mcts_simulations": 500,
  "mcts_depth": 5,
  "output_dir": "./results"
}
```

**Validation:** The sum of agents in `agent_composition` must equal `players`. If not, exit with error showing the mismatch.

### 3.4 Conversation Log Extraction

For every game, save a complete conversation log to `<output_dir>/game_<id>/conversation.txt`:

```
=== GAME 42 — 8 PLAYERS ===
Roles (revealed at end):
  P1: Villager (HeuristicAgent)
  P2: Werewolf (BayesianAgent)
  P3: Villager (BayesianAgent+Suspicion)
  ...

--- DAY 1 ---
[Round 1]
P1: VOTE P5
P2: ESTIMATE P3 WEREWOLF
P3: COMINGOUT P3 SEER
P3: DIVINED P2 WEREWOLF
...

[Voting]
P1 → P5
P2 → P3
P3 → P2
...
Eliminated: P5 (was Villager)

--- NIGHT 1 ---
Seer P3 divined P4 → HUMAN
Werewolf P2 attacked P4
Bodyguard P7 protected P3 → attack failed

--- DAY 2 ---
...

=== RESULT: VILLAGE WINS in 4 days ===
```

### 3.5 Suspicion Agent Reasoning Trace

For every game where a suspicion-enhanced agent participates, save a detailed reasoning trace to `<output_dir>/game_<id>/suspicion_trace.txt`:

```
=== SUSPICION AGENT TRACE: P3 (BayesianAgent+Suspicion) ===
True role: Villager

--- DAY 1, END OF ROUND 5 ---

Bayesian beliefs P(p = wolf):
  P1: 0.20    P2: 0.45    P4: 0.30    P5: 0.10
  P6: 0.25    P7: 0.15    P8: 0.55

Suspicion scores σ(p):
  P1: 0.50    P2: 0.65    P4: 0.50    P5: 0.40
  P6: 0.55    P7: 0.50    P8: 0.70

Detector activations this round:
  P2:
    D1 (vote-accusation mismatch): 0.50  → P2 accused P3 but voted P5
    D2 (bandwagon): 0.00            → independent vote
    D3 (pressure-trigger): 0.20     → mild correlation
    D4 (protection):       0.00     → no consistent shield yet
    D5 (claim consistency): 0.00    → no claims made
    Composite E(P2): 0.19           → suspicion DECREASED slightly

  P8:
    D1: 0.66                        → multiple mismatches
    D2: 0.50                        → followed majority twice
    D3: 0.60                        → strong pressure-triggered pattern
    D4: 0.00                        
    D5: 1.00                        → claimed Seer but contradicted
    Composite E(P8): 0.62           → suspicion INCREASED to 0.79

Combined decision scores: score(p) = 0.6*belief + 0.4*σ
  P1: 0.32    P2: 0.53    P4: 0.38    P5: 0.22
  P6: 0.37    P7: 0.29    P8: 0.61

DECISION: Vote P8 (highest combined score)
RATIONALE:
  - Highest belief that P8 is wolf: 0.55
  - Highest suspicion: 0.79 (driven by D5 claim contradiction and D1 mismatches)
  - Combined score 0.61 well above next-highest (P2 at 0.53)
  - D5 trigger is strong evidence: P8 claimed Seer, but their claimed
    divine result on P3 (called wolf) contradicts P3's actual role (villager)

--- DAY 2, END OF ROUND 5 ---
...
```

This trace is **the most important deliverable** for the research paper — it provides the qualitative evidence to interpret quantitative results.

### 3.6 Per-Round Suspicion/Trust Evolution Log

For every game with a suspicion agent, save a CSV at `<output_dir>/game_<id>/suspicion_evolution.csv`:

```csv
day,round,target_player,target_role,sigma,trust,belief_wolf,combined_score,d1,d2,d3,d4,d5,E,delta_sigma
1,1,P1,Villager,0.50,0.50,0.20,0.32,0.00,0.00,0.00,0.00,0.00,0.00,0.00
1,1,P2,Werewolf,0.50,0.50,0.30,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00
1,5,P2,Werewolf,0.65,0.35,0.45,0.53,0.50,0.00,0.20,0.00,0.00,0.19,-0.01
1,5,P8,Werewolf,0.79,0.21,0.55,0.61,0.66,0.50,0.60,0.00,1.00,0.62,0.10
...
```

Where `trust = 1 - σ`. This data enables time-series plots showing how suspicion evolves through the game and which detectors triggered when.

### 3.7 Suspicion Agent Picker

**At the start of each game**, the suspicion-enhanced agent is assigned to a specific player slot. The user can configure this via `--suspicion-agent` (e.g. `bayesian` means a Bayesian agent enhanced with suspicion is added to the game).

The implementation must:
1. Identify which player slot holds the suspicion agent (random assignment by default)
2. Force the suspicion agent's role to NEVER be Werewolf (see 3.9)
3. Log the suspicion agent's player ID and assigned role at game start

### 3.8 Grid Search Mode

When `--grid-search` flag is set, run the systematic two-stage hyperparameter search:

**Stage 1: Detector weights (a1-a5)**
- Vary each weight from 0.0 to 0.5 in steps of 0.1
- Constraint: weights must sum to 1.0 (normalize after sampling)
- Use Latin hypercube sampling to keep total combinations manageable (~200 configs)
- Run 2,000 games per configuration

**Stage 2: Decision weights (w1, w2)**
- Using best detector weights from Stage 1
- Vary w1 from 0.0 to 1.0 in steps of 0.1, w2 = 1 - w1
- Run 2,000 games per configuration (11 configs total)

Output a CSV at `<output_dir>/grid_search_results.csv`:
```csv
config_id,a1,a2,a3,a4,a5,w1,w2,games,village_winrate,wolf_id_precision,suspicion_calibration
0,0.30,0.15,0.20,0.20,0.15,0.6,0.4,2000,0.583,0.412,0.621
1,0.25,0.20,0.20,0.20,0.15,0.6,0.4,2000,0.571,0.398,0.605
...
```

Print the top 5 configurations by village win rate at the end.

### 3.9 Suspicion Agent Cannot Be Werewolf

**Hard constraint:** The suspicion-enhanced agent must never be assigned the Werewolf role.

Implementation in role assignment:
1. First, randomly select the suspicion agent's player slot (or use configured slot)
2. Assign Werewolf role to wolf-slots, **excluding** the suspicion agent's slot
3. Assign remaining roles (Villager, Seer, Possessed, Bodyguard) randomly among other slots
4. The suspicion agent receives a non-Werewolf role with the appropriate distribution

**Why:** The suspicion module is designed to detect deception in others. Having it play as a wolf would require it to use itself for deception, which conflates the research question. Restricting to non-wolf roles keeps the experimental design clean.

**Note:** The suspicion agent CAN be Possessed (wolf team but not literally a wolf). Possessed is a unique case — they're on the wolf team but appear human to the Seer. Allow this; it tests whether the suspicion module correctly handles the agent's own deceptive position.

**Logging:** Always record the suspicion agent's actual role at game start so analysis can stratify results by role (e.g., "win rate when suspicion agent was Villager vs Seer").

---

## 4. File Structure

```
Adv-AI-Project-v2/
├── run_game.py              # CLI entry point with argparse
├── werewolf/
│   ├── __init__.py
│   ├── game.py              # WerewolfGame class
│   ├── gameinfo.py          # GameInfo, Talk, Vote, Judge
│   ├── const.py             # Role, Status, Species enums
│   ├── role_assigner.py     # Handles 3.9 constraint
│   ├── agent.py             # AbstractAgent base class
│   └── agents/
│       ├── random_agent.py
│       ├── heuristic_agent.py
│       ├── bayesian_agent.py
│       ├── mcts_agent.py
│       └── logic_agent.py
├── suspicion/
│   ├── __init__.py
│   ├── module.py            # SuspicionModule class
│   ├── detectors.py         # D1-D5 implementations
│   └── enhanced_agent.py    # Wraps any compatible agent + module
├── runner/
│   ├── __init__.py
│   ├── game_runner.py       # Runs N games, manages composition
│   ├── grid_search.py       # Two-stage grid search
│   ├── conversation_logger.py   # 3.4
│   ├── suspicion_tracer.py      # 3.5
│   └── evolution_logger.py      # 3.6
├── configs/
│   └── default.json
├── results/                 # Auto-created output folder
└── tests/
    ├── test_game.py
    ├── test_agents.py
    └── test_suspicion.py
```

---

## 5. PowerShell Usage Examples

```powershell
# Default 100-game run with 8 players, Bayesian+Suspicion
python run_game.py

# 1000 games with 5 players, MCTS+Suspicion
python run_game.py --games 1000 --players 5 --suspicion-agent mcts

# Custom composition via JSON config
python run_game.py --config configs/exp1.json

# Grid search for optimal weights (long-running)
python run_game.py --grid-search --players 8 --output ./grid_results

# Reproducible run with fixed seed
python run_game.py --games 500 --seed 42 --verbose

# Baseline comparison (no suspicion)
python run_game.py --games 1000 --suspicion-agent none
```

---

## 6. Output Directory Structure

After a run, the output directory contains:

```
results/
├── summary.csv                      # Overall results across all games
├── config_used.json                 # Echo of the configuration
├── game_0001/
│   ├── conversation.txt             # 3.4
│   ├── suspicion_trace.txt          # 3.5 (only if suspicion agent present)
│   ├── suspicion_evolution.csv      # 3.6
│   └── game_summary.json            # Per-game metadata
├── game_0002/
│   └── ...
└── grid_search_results.csv          # Only if --grid-search was used
```

`summary.csv` columns:
```
game_id, n_players, winner, n_days, suspicion_agent_id, suspicion_agent_role,
suspicion_agent_survived, suspicion_agent_won,
correct_wolf_votes, total_votes, false_accusations
```

---

## 7. Implementation Priorities

Build in this order:

1. **Core game engine** (`werewolf/game.py`, `gameinfo.py`, `const.py`) — get a basic game running with 5 players
2. **Random and Heuristic agents** — verify game mechanics work
3. **Role assigner with suspicion-not-wolf constraint** (3.9)
4. **Bayesian agent** — most important baseline
5. **Suspicion module** — separate class, plug into Bayesian first
6. **CLI with all args** (3.1, 3.2, 3.3, 3.7) — make it runnable from PowerShell
7. **Logging systems** (3.4, 3.5, 3.6) — every game produces all three logs
8. **MCTS agent** — implement after core flow works
9. **Logic-Based agent** — implement last (most complex)
10. **Grid search mode** (3.8) — final feature

---

## 8. Testing Requirements

Before running large experiments, verify:
- Random agent vs Random agent: village win rate should be ~50%
- Bayesian agent should beat Heuristic agent in head-to-head (>55% win rate)
- Suspicion-not-wolf constraint holds across 1,000 random role assignments
- All three log files generate correctly for every game
- CLI arg validation rejects invalid player counts and bad compositions
- Grid search with reduced game count (50 games per config) completes in under 1 hour

---

## 9. Performance Targets

- 8-player game: <10 seconds per game
- 10,000 games (8-player): <12 hours on standard laptop
- Grid search Stage 1 (200 configs × 2000 games): can run overnight
- Memory: <500MB peak during normal run

---

*This is the build specification. Claude Code should implement everything described here, in the priority order given. All numerical parameters (weights, simulation counts, etc.) must be configurable, never hardcoded. Every game must produce its three log files.*
