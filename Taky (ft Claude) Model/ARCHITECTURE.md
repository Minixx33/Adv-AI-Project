# Suspicion-Aware Werewolf AI — Architecture Document
## For use as context when building the self-contained game engine

---

## Project Goal

Build a self-contained Python Werewolf game engine that:
- Mirrors the AIWolf platform's agent structure exactly (same method names, same GameInfo structure)
- Runs without Java server or socket connections
- Implements 4 agent types for comparison
- Logs all game data to CSV for analysis

---

## Game Rules (5-Player Setup)

**Roles:**
- Villager (2 players) — village team, no special ability
- Werewolf (1 player) — wolf team, kills at night
- Seer (1 player) — village team, can divine one player's role per night
- Possessed (1 player) — wolf team, appears as villager to Seer

**Win conditions:**
- Village team wins when all werewolves are eliminated
- Wolf team wins when wolves equal or outnumber villagers

**Game flow:**
```
Day 0: Roles assigned, initialize agents
↓
Day Phase: Agents talk (5 rounds of discussion), then vote to eliminate
↓
Night Phase: Seer divines, Werewolf attacks
↓
Repeat until win condition met
```

**Talk protocol (mirrors AIWolf):**
- VOTE {target} — declare intention to vote
- ESTIMATE {target} {role} — claim belief about player's role
- COMINGOUT {target} {role} — claim a role
- DIVINED {target} {result} — Seer shares divine result
- SKIP — pass this talk turn
- OVER — done talking this round

---

## AIWolf Agent Interface (must mirror exactly)

Every agent implements these methods:

```python
class AbstractAgent:
    def initialize(self, game_info, game_setting):
        """Called once at game start. Set up all data structures here."""
        pass
    
    def update(self, game_info):
        """Called before every other method. Receive new GameInfo."""
        pass
    
    def day_start(self):
        """Called at start of each day phase."""
        pass
    
    def talk(self):
        """Called during discussion phase. Return a talk string."""
        return "Skip"
    
    def vote(self):
        """Called during voting phase. Return target Agent."""
        pass
    
    def whisper(self):
        """Werewolf only — night communication between wolves."""
        return "Skip"
    
    def attack(self):
        """Werewolf only — choose who to kill at night."""
        pass
    
    def divine(self):
        """Seer only — choose who to investigate."""
        pass
    
    def finish(self):
        """Called when game ends."""
        pass
```

---

## GameInfo Structure (mirrors AIWolf)

```python
@dataclass
class GameInfo:
    day: int                          # Current day number
    agent: Agent                      # This agent's identity
    role_map: Dict[Agent, Role]       # Only contains this agent's own role
    status_map: Dict[Agent, Status]   # ALIVE or DEAD for all players
    talk_list: List[Talk]             # All talks this round
    whisper_list: List[Talk]          # Werewolf whispers (wolf team only)
    vote_list: List[Vote]             # Votes cast this round
    attack_vote_list: List[Vote]      # Wolf attack votes (wolf team only)
    executed_agent: Optional[Agent]   # Who was eliminated today
    attacked_agent: Optional[Agent]   # Who was attacked last night
    divine_result: Optional[Judge]    # Seer's result (seer only)
    alive_agent_list: List[Agent]     # All alive agents
    last_dead_agent_list: List[Agent] # Agents who died this round
    
@dataclass    
class Talk:
    day: int
    agent: Agent
    text: str
    
@dataclass
class Vote:
    day: int
    agent: Agent      # who voted
    target: Agent     # who they voted for
    
@dataclass
class Judge:
    day: int
    agent: Agent      # the seer
    target: Agent     # who was divined
    result: Species   # HUMAN or WEREWOLF
```

---

## The 4 Agents

### Agent 1: Random Agent (Baseline 1)
```
- vote(): random.choice(alive_players excluding self)
- talk(): always return "Skip"
- attack(): random.choice(alive_villagers)
- divine(): random.choice(alive_players excluding self)
- No belief tracking
```

### Agent 2: Heuristic Agent (Baseline 2)
```
- Tracks accusation_count[player] — how many times each player has been accused
- vote(): argmax(accusation_count) among alive players
- talk(): VOTE {most_accused_player}
- attack(): target player with lowest accusation count (least suspicious = most dangerous)
- divine(): target player with highest accusation count
```

### Agent 3: Bayesian Agent (Baseline 3)
```
Layer 1 only — no suspicion layer

State: belief[player] = P(player is Werewolf) for all other players
Initialize: belief[p] = 1/4 for all p (uniform prior, 1 wolf among 4 others)

Update after each observation:
- Someone voted for p → slight increase in belief[p] (wolves vote for villagers)
- Someone defended p → slight decrease in belief[p]  
- Seer result on p → set belief[p] = 1.0 or 0.0 with certainty
- p was eliminated and was wolf → update all beliefs (one less wolf)
- p was eliminated and was villager → update all beliefs

vote(): argmax(belief) among alive players
talk(): VOTE {argmax(belief)}
divine(): target argmax(belief) — investigate most suspicious
attack(): target argmin(belief) — kill least suspicious (most trusted villager)
```

### Agent 4: Suspicion-Aware Agent (YOUR CONTRIBUTION)
```
Layer 1: Bayesian role belief (same as Agent 3)
  belief[p] = P(p is Werewolf)

Layer 2: Suspicion score
  sigma[p] ∈ [0, 1] — initialized to 0.5 for all players

Five behavioral pattern detectors:

D1 - Vote-Accusation Mismatch:
  Track if player accused X but voted for Y ≠ X
  D1[p] = mismatches[p] / rounds_played[p]

D2 - Bandwagon Index:
  Track if player always follows majority vote
  D2[p] = times_followed_majority[p] / applicable_rounds[p]

D3 - Pressure-Triggered Accusations:
  Detect if player accuses others when they themselves are under suspicion
  D3[p] = correlation(suspicion_on_p, accusations_by_p)

D4 - Protection Patterns:
  Detect if player consistently defends a specific other player
  D4[p] = max over q of (times_p_defended_q / rounds)

D5 - Claim Consistency:
  Track if role claims are contradicted by game outcomes
  D5[p] = contradicted_claims[p] / total_claims[p]

Composite evidence:
  E[p] = 0.30*D1[p] + 0.15*D2[p] + 0.20*max(D3[p],0) + 0.20*D4[p] + 0.15*D5[p]

Suspicion update (asymmetric — faster to lose trust than gain it):
  if E[p] > 0.5:
      sigma[p] = min(1.0, sigma[p] + 0.15 * E[p])   # fast increase
  else:
      sigma[p] = max(0.0, sigma[p] - 0.05 * (1-E[p])) # slow decrease

Combined decision score:
  score[p] = 0.6 * belief[p] + 0.4 * sigma[p]

vote(): argmax(score) among alive players
talk(): VOTE {argmax(score)}
divine(): argmax(score) — investigate most suspicious
attack(): argmin(score) — kill least suspicious
```

---

## Game Engine Structure

```python
class WerewolfGame:
    def __init__(self, agents, roles):
        # assigns roles, initializes game state
    
    def run(self):
        # runs complete game, returns winner
        # Day 0: initialize
        # Loop: day_phase() → night_phase() → check_winner()
    
    def day_phase(self):
        # discussion rounds (5 talks per agent)
        # voting → elimination
    
    def night_phase(self):
        # seer divines
        # werewolf attacks
    
    def check_winner(self):
        # returns "village" or "wolf" or None
    
    def build_game_info(self, agent):
        # constructs GameInfo from each agent's perspective
        # IMPORTANT: role_map only contains the requesting agent's own role
```

---

## Game Runner & Logging

```python
class GameRunner:
    def run_experiments(self, n_games=10000):
        # runs N games, rotates agent positions
        # logs all results to CSV
    
    def log_game(self, game_result):
        # appends to results.csv:
        # game_id, winner, n_rounds, 
        # agent_type, role, survived, votes_correct,
        # suspicion_accuracy (for suspicion agent)
```

**CSV output columns:**
```
game_id, winner, n_rounds, agent_name, agent_role, survived, 
votes_cast, correct_wolf_votes, false_accusations,
final_belief_wolf (bayesian/suspicion agents only),
final_sigma_wolf (suspicion agent only)
```

---

## File Structure to Create

```
Adv-AI-Project/
├── werewolf/
│   ├── __init__.py
│   ├── game.py          # WerewolfGame engine
│   ├── gameinfo.py      # GameInfo, Talk, Vote, Judge dataclasses
│   ├── const.py         # Role, Status, Species enums
│   ├── agent.py         # AbstractAgent base class
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── bayesian_agent.py
│   │   └── suspicion_agent.py
│   └── runner.py        # GameRunner + CSV logging
├── run_experiments.py   # main entry point
└── results/             # CSV output folder
```

---

## Key Implementation Notes

1. **Role assignment** must be random each game — rotate which agent type gets which role
2. **GameInfo** must be constructed from each agent's perspective — wolves see wolf team info, seer sees divine results, villagers see neither
3. **Talk parsing** — agents need to parse other agents' talk strings to extract VOTE, ESTIMATE, COMINGOUT, DIVINED intentions
4. **Voting resolution** — plurality vote, random tiebreak
5. **Night phase order** — Seer divines first, then wolf attacks. If wolf attacks the seer, seer still gets their result for that night
6. **Suspicion detectors** need at least 2 rounds of data before being meaningful — initialize with neutral values

---

## Evaluation Metrics to Track

| Metric | Formula |
|--------|---------|
| Win rate | wins / total_games (per agent type, per role) |
| Wolf ID precision | correct_wolf_votes / total_votes |
| Wolf ID recall | wolves_voted_against / total_wolves |
| Survival rate | rounds_survived / total_rounds |
| False accusation rate | innocent_accusations / total_accusations |
| Suspicion calibration | correlation(sigma_wolf, true_wolf_status) |

---

*MLR 555 — Advanced Artificial Intelligence*
*Suspicion-Aware Agent for Social Deduction Games (Werewolf)*
