# Werewolf AI v2 — Update Specification

## Context for Claude Code

This document describes **targeted updates** to the existing v2 Werewolf project. Do **NOT** rebuild the project from scratch. Read the existing codebase first, understand the current architecture, then make incremental changes as specified below.

The current v2 has the basic structure working (5 agents, CLI, logging, suspicion module), but lacks rich conversational behavior. This update adds an expanded talk protocol so agents have meaningful back-and-forth conversation, which feeds the suspicion detectors with much richer behavioral data.

---

## What to Change

### Change 1: Expand the Talk Protocol

**Location:** `werewolf/gameinfo.py` (or wherever Talk tokens are defined) and the agent talk methods.

**Current state:** Agents likely emit only basic tokens (VOTE, ESTIMATE, COMINGOUT, DIVINED, SKIP, OVER) and only 1-2 per round.

**New requirement:** Add the following talk tokens to the protocol:

| Token | Meaning | Example |
|-------|---------|---------|
| `AGREE Pn` | Agrees with Pn's last statement | `AGREE P3` |
| `DISAGREE Pn` | Disagrees with Pn's last statement | `DISAGREE P5` |
| `REQUEST Pn action` | Asks Pn to do something | `REQUEST P3 reveal_role` |
| `INQUIRE Pn` | Questions Pn's behavior or claim | `INQUIRE P5` |
| `BECAUSE reason` | Justification for previous statement | `BECAUSE seer_claim_conflict` |
| `ATTACK Pn` | Strong public accusation | `ATTACK P5` |
| `DEFEND Pn` | Defends Pn against accusations | `DEFEND P3` |
| `NOT Pn role` | Asserts Pn is NOT a particular role | `NOT P3 SEER` |
| `IDENTIFIED Pn role` | Medium identifies a dead player's role | `IDENTIFIED P5 WEREWOLF` |
| `GUARDED Pn` | Bodyguard claims to have protected Pn | `GUARDED P3` |

Keep all existing tokens (VOTE, ESTIMATE, COMINGOUT, DIVINED, SKIP, OVER).

**BECAUSE reason codes** (standardized for parseability):
- `divine_result`
- `seer_claim_conflict`
- `vote_pattern`
- `bandwagon`
- `defended_wolf`
- `claim_contradiction`
- `silent_player`
- `aggressive_accuser`
- `protect_seer`
- `last_resort`

### Change 2: Increase Talk Budget Per Round

**Location:** Wherever the game loop calls `agent.talk()` per round.

**Current state:** Likely calls `talk()` once per agent per round.

**New requirement:** Each agent gets a budget of **up to 5 talk tokens per round**. The game cycles through alive players in randomized order, calling `talk()` repeatedly until either:
- The agent emits `OVER` (done talking this round)
- The agent has used 5 tokens
- The agent emits `SKIP` (declines to speak this turn — moves to next agent immediately)

The total round talk-list should grow to ~30 tokens for an 8-player game (vs. ~8 currently).

### Change 3: Update Agent Talk Strategies

**Location:** Each agent's `talk()` method in `werewolf/agents/`.

Each agent needs richer talk behavior. Update them as follows:

**Random Agent** (`random_agent.py`):
- Emits 1-2 random tokens per round
- Most often VOTE for random target, or SKIP
- No need to be sophisticated — keep simple

**Heuristic Agent** (`heuristic_agent.py`):
- Token 1: VOTE for current most-accused player
- Token 2: AGREE Pn if another player attacked the same target
- Token 3: ATTACK on the most-accused player
- Token 4: OVER

**Bayesian Agent** (`bayesian_agent.py`):
- Token 1: ESTIMATE Pn WEREWOLF for top wolf candidate (highest belief)
- Token 2: ATTACK Pn for the same target
- Token 3: BECAUSE with appropriate reason code (e.g., `vote_pattern` if based on voting history, `divine_result` if seer-derived)
- Token 4: VOTE Pn for the same target
- Token 5: OVER
- If Bayesian agent is Seer: emit COMINGOUT + DIVINED early

**MCTS Agent** (`mcts_agent.py`):
- Run MCTS to determine best vote target (existing logic)
- Token 1: ESTIMATE Pn WEREWOLF for that target
- Token 2: VOTE Pn for that target
- Token 3: AGREE/DISAGREE on previous round's dominant accusation based on whether it matches MCTS's preferred target
- Token 4: OVER

**Logic-Based Agent** (`logic_agent.py`):
- If Seer: COMINGOUT + DIVINED
- For each player whose claim contradicts derived facts: emit NOT Pn role
- For inconsistent claimants: emit INQUIRE Pn
- VOTE for highest-derived-suspicion player
- BECAUSE claim_contradiction or seer_claim_conflict

### Change 4: Update Detector Implementations

**Location:** `suspicion/detectors.py`

Each detector now consumes the new tokens directly. Update the implementations:

**D1 — Vote-Accusation Mismatch:**
```python
# Compare ATTACK statements vs actual votes
# For each ATTACK token by player p targeting Pn,
# check if p's actual vote was Pn
# If not: count as mismatch
mismatches = count of (ATTACK by p targeting X) where p.vote != X
D1(p) = mismatches / total_attacks_by(p)
```

**D2 — Bandwagon Index:**
```python
# Count AGREE tokens emitted after a majority has formed
# AGREE Pn after Pn has emitted ATTACK Q where Q already has 2+ accusers
bandwagon = count of AGREE tokens by p where the agreed-with statement was an ATTACK on already-targeted player
D2(p) = bandwagon / total_agree_by(p)
```

**D3 — Pressure-Triggered Accusations:**
```python
# Per round r, compute:
#   pressure(p, r) = count of (INQUIRE p) + (ATTACK p) by other players in round r
#   aggression(p, r) = count of (ATTACK Pn) tokens emitted by p in round r
# Then D3(p) = pearson_correlation(pressure_series, aggression_series)
```

**D4 — Protection Patterns:**
```python
# For each pair (p, q):
#   defends = count of (DEFEND q) + (DISAGREE r) where r attacked q
#                    + (NOT q WEREWOLF) tokens emitted by p
# D4(p) = max over q of (defends(p, q) / total_rounds)
```

**D5 — Claim Consistency:**
```python
# Track all COMINGOUT and DIVINED tokens
# When game ends or a player is revealed:
#   For each COMINGOUT P role by claimant c:
#     If actual role of P != claimed role: contradicted += 1
#   For each DIVINED P result by claimant c:
#     If actual species of P contradicts result: contradicted += 1
# D5(c) = contradicted(c) / total_claims_by(c)
```

### Change 5: Update Conversation Log Format

**Location:** `runner/conversation_logger.py`

The log file `conversation.txt` currently shows just votes and basic actions. Update to show the **full token stream** for each round:

```
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
P5: ATTACK P3
P5: OVER

P3: INQUIRE P5
P3: NOT P3 WEREWOLF
P3: ESTIMATE P5 WEREWOLF
P3: BECAUSE seer_claim_conflict
P3: OVER

(... continues for all alive agents ...)

[Voting Phase — Day 1]
P1 → P5     P2 → P3     ...
Tally: P5(5), P3(2), P6(1)
Eliminated: P5 (was Werewolf)
```

### Change 6: Update Suspicion Trace to Reference Specific Tokens

**Location:** `runner/suspicion_tracer.py`

Currently the trace shows detector values but not which specific tokens triggered them. Update each detector's evidence section to cite the actual talk tokens:

```
P8 — Composite E(P8): 0.62 → suspicion INCREASED to 0.79
  D1 (vote-accusation mismatch): 0.66
    Evidence: P8 emitted "ATTACK P6" in Rounds 2,3 but never voted P6
              Voted P5 instead
  D2 (bandwagon): 0.50
    Evidence: P8 emitted "AGREE P5" in Round 1 (joined existing accusation)
              Then "AGREE P6" in Round 4 (joined new majority)
  D3 (pressure-trigger): 0.60
    Evidence: After P3 emitted "INQUIRE P8" in Round 3, P8 emitted
              ATTACK tokens against P3 in Rounds 4 and 5
  D5 (claim consistency): 1.00
    Evidence: P8 emitted "COMINGOUT P8 SEER" + "DIVINED P3 WEREWOLF"
              Contradicts P6's verified Seer claim
```

The structure stays the same — just enrich the "Evidence:" lines with specific token references.

### Change 7: Add Wolf Talk Strategies

**Location:** Werewolf agents (likely in `werewolf/agents/bayesian_agent.py` when role is WEREWOLF, or wherever wolf behavior is conditional)

Add these wolf-specific talk strategies as configurable behaviors:

**Bus-Driver Wolf** (probability 0.2 per game):
- 20% of rounds: emit ATTACK toward fellow wolf
- Other rounds: ATTACK on a non-wolf target

**False-Claimer Wolf** (probability 0.5 per game):
- Round 1: COMINGOUT WEREWOLF self as SEER
- Round 1: DIVINED random non-wolf villager as WEREWOLF
- Subsequent rounds: defend the false Seer claim

These should be enabled/disabled via config in the JSON file:
```json
{
  "wolf_strategies": {
    "bus_driver_probability": 0.2,
    "false_claimer_probability": 0.5
  }
}
```

---

## What NOT to Change

- The CLI argument structure (`--games`, `--players`, `--suspicion-agent`, `--config`, `--grid-search`, etc.)
- The role assigner's "suspicion agent cannot be wolf" constraint
- The grid search logic
- The output directory structure
- The summary.csv format
- The CSV evolution log format (suspicion_evolution.csv) — same columns, just richer data flowing in
- The asymmetric suspicion update rule
- The combined decision score formula `score(p) = w1 * base_belief(p) + w2 * σ(p)`

---

## Testing After Update

Verify each of these:

1. **Talk tokens parse correctly:** Run a single game and check `conversation.txt` shows ~30 tokens per round
2. **Detectors fire:** Run a game with a Bus-Driver wolf — D1 and D4 should activate on that wolf
3. **Detectors fire on False-Claimer:** Run with a False-Claimer wolf — D5 should activate by mid-game
4. **Win rates didn't crash:** Run 100 games comparing baseline (no suspicion) vs Bayesian+Suspicion. Both should produce reasonable results (no errors, win rates between 30-70%)
5. **Suspicion trace cites tokens:** The trace file should reference specific tokens like "P8 emitted ATTACK P3 in Round 4" rather than just numerical values

---

## Implementation Order

Apply these changes in order — each step's changes can be tested before moving to the next:

1. **Change 1** (expand talk tokens) — additive, won't break existing code
2. **Change 2** (increase talk budget) — modify game loop
3. **Change 3** (update agent talk strategies) — one agent at a time, test each
4. **Change 5** (conversation log format) — visible in output, easy to verify
5. **Change 4** (update detectors) — most important for research validity
6. **Change 6** (enriched suspicion trace) — depends on Change 4
7. **Change 7** (wolf strategies) — final addition, used in Experiment 4

After step 3, you should already see richer conversations in the logs even before the detectors are updated. After step 5, the detectors should produce noticeably different (better calibrated) suspicion scores.

---

*This is an update specification, not a rebuild spec. Preserve all existing working code and add only what is described here.*
