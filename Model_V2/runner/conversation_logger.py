"""
runner/conversation_logger.py

Formats a game record's event list into the human-readable conversation log
specified in §3.4 of the architecture (updated for v2 rich-talk protocol).

Hierarchy / contents:
  ConversationLogger
    └── write(game_record, output_dir)
          Writes <output_dir>/game_<id>/conversation.txt

Output format (v2 update):
  === GAME 42 — 8 PLAYERS ===
  Roles (revealed at end):
    P1: Villager (HeuristicAgent)
    ...

  --- DAY 1 ---
  [Round 1]
  P6: COMINGOUT P6 SEER
  P6: DIVINED P5 WEREWOLF
  P6: BECAUSE divine_result
  P6: VOTE P5
  P6: Over

  P5: DISAGREE P6
  P5: ATTACK P3
  P5: Over

  (... all alive agents ...)

  [Voting Phase — Day 1]
  P1 → P5    P2 → P3    ...
  Tally: P5(5), P3(2)
  Eliminated: P5 (was WEREWOLF)

  --- NIGHT 1 ---
  Seer P3 divined P4 → HUMAN
  Werewolf P2 attacked P4
  ...

  === RESULT: VILLAGE WINS in 4 days ===
"""

import os
from collections import Counter
from io import StringIO


class ConversationLogger:
    """Writes conversation.txt from a GameRecord dict."""

    def write(self, game_record: dict, output_dir: str) -> str:
        gid    = game_record["game_id"]
        folder = os.path.join(output_dir, f"game_{gid:04d}")
        os.makedirs(folder, exist_ok=True)
        path   = os.path.join(folder, "conversation.txt")

        text = self._render(game_record)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(self, rec: dict) -> str:
        buf = StringIO()
        n   = rec["n_players"]
        gid = rec["game_id"]

        buf.write(f"=== GAME {gid} — {n} PLAYERS ===\n")
        buf.write("Roles (revealed at end):\n")
        for pid in sorted(rec["roles"]):
            buf.write(f"  P{pid}: {rec['roles'][pid]} ({rec['agent_names'].get(pid, '?')})\n")
        buf.write("\n")

        events = rec["events"]
        i = 0
        day = None

        while i < len(events):
            ev    = events[i]
            etype = ev.get("type")

            if etype == "day_start":
                day = ev["day"]
                buf.write(f"--- DAY {day} ---\n")
                i += 1

            elif etype == "talk":
                rnd = ev.get("round", 0)
                # Collect all talk events for this round on this day
                j = i
                round_talks = []
                while (j < len(events)
                       and events[j].get("type") == "talk"
                       and events[j].get("round") == rnd
                       and events[j].get("day") == day):
                    round_talks.append(events[j])
                    j += 1

                buf.write(f"[Round {rnd + 1}]\n")
                # Group consecutive tokens by speaker
                k = 0
                while k < len(round_talks):
                    speaker = round_talks[k]["agent"]
                    # Emit all consecutive tokens from this speaker
                    while k < len(round_talks) and round_talks[k]["agent"] == speaker:
                        t = round_talks[k]
                        buf.write(f"P{t['agent']}: {t['text']}\n")
                        k += 1
                    buf.write("\n")
                i = j

            elif etype == "vote":
                # Collect all votes for this day
                votes = []
                j = i
                while (j < len(events)
                       and events[j].get("type") == "vote"
                       and events[j].get("day") == day):
                    votes.append(events[j])
                    j += 1

                buf.write(f"[Voting Phase — Day {day}]\n")
                # Vote arrows: 4 per line
                for start in range(0, len(votes), 4):
                    chunk = votes[start:start + 4]
                    buf.write("  " + "    ".join(
                        f"P{v['agent']} -> P{v['target']}" for v in chunk
                    ) + "\n")

                # Tally
                tally = Counter(v["target"] for v in votes)
                tally_str = ", ".join(
                    f"P{pid}({cnt})"
                    for pid, cnt in sorted(tally.items(), key=lambda x: -x[1])
                )
                buf.write(f"Tally: {tally_str}\n")
                i = j

            elif etype == "execution":
                buf.write(f"Eliminated: P{ev['agent']} (was {ev['role']})\n\n")
                i += 1

            elif etype == "night_start":
                buf.write(f"--- NIGHT {ev['day']} ---\n")
                i += 1

            elif etype == "divine":
                buf.write(
                    f"Seer P{ev['seer']} divined P{ev['target']} -> {ev['result']}\n"
                )
                i += 1

            elif etype == "guard":
                buf.write(
                    f"Bodyguard P{ev['bodyguard']} protected P{ev['target']}\n"
                )
                i += 1

            elif etype == "whisper":
                i += 1  # omit whispers from public log

            elif etype == "attack":
                votes = ev.get("votes", {})
                if votes:
                    buf.write(
                        "Wolves attacked: "
                        + ", ".join(f"P{k}->P{v}" for k, v in votes.items())
                        + "\n"
                    )
                i += 1

            elif etype == "guard_saved":
                buf.write(f"Bodyguard saved P{ev['target']} — attack failed\n")
                i += 1

            elif etype == "death":
                buf.write(f"Killed overnight: P{ev['agent']} (was {ev['role']})\n\n")
                i += 1

            elif etype == "game_end":
                winner = ev["winner"]
                n_days = ev.get("n_days", "?")
                buf.write(f"\n=== RESULT: {winner} WINS in {n_days} days ===\n")
                i += 1

            else:
                i += 1

        return buf.getvalue()
