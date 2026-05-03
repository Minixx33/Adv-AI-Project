"""
runner/xlsx_exporter.py

Combines the conversation log and suspicion trace into a single .xlsx file
with two sheets per game:

  "Conversation"     — structured event log (Day / Phase / Round / Agent / Role / Text)
  "Suspicion Trace"  — one row per (snapshot × alive player) with all detector values

Replaces the separate conversation.txt and suspicion_trace.txt outputs.
"""

import os

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ── Colour palette ──────────────────────────────────────────────────────────
_FILL_HEADER    = PatternFill("solid", fgColor="2F5496")   # dark blue
_FILL_DAY       = PatternFill("solid", fgColor="D9E1F2")   # light blue
_FILL_NIGHT     = PatternFill("solid", fgColor="D6DCE4")   # grey
_FILL_VOTE      = PatternFill("solid", fgColor="FFF2CC")   # light yellow
_FILL_EXECUTION = PatternFill("solid", fgColor="FCE4D6")   # light orange
_FILL_RESULT    = PatternFill("solid", fgColor="E2EFDA")   # light green
_FILL_DECISION  = PatternFill("solid", fgColor="FFC7CE")   # light red (voted target)

_FONT_HEADER = Font(bold=True, color="FFFFFF")
_FONT_SECTION = Font(bold=True)


def _role_str(r) -> str:
    """Convert a Role enum or plain string to a display string."""
    if r is None:
        return ""
    return r.value if hasattr(r, "value") else str(r)


class XlsxExporter:
    """
    Writes <output_dir>/game_<id>/game_<id>.xlsx containing:
      Sheet 1 — Conversation (always present)
      Sheet "Trace P{id}" — one per suspicion agent with non-empty snapshots
    """

    def write(
        self,
        game_record:      dict,
        suspicion_agents: list,   # [{agent_id, agent_name, trace_snapshots, decision_weights}]
        true_roles:       dict,
        output_dir:       str,
    ) -> str:
        gid    = game_record["game_id"]
        folder = os.path.join(output_dir, f"game_{gid:04d}")
        os.makedirs(folder, exist_ok=True)
        path   = os.path.join(folder, f"game_{gid:04d}.xlsx")

        wb = openpyxl.Workbook()
        self._write_conversation(wb, game_record)
        for agent_data in suspicion_agents:
            if agent_data.get("trace_snapshots"):
                self._write_suspicion_trace(wb, agent_data, true_roles)

        wb.save(path)
        return path

    # ------------------------------------------------------------------ #
    #  Sheet 1 — Conversation                                             #
    # ------------------------------------------------------------------ #

    def _write_conversation(self, wb: openpyxl.Workbook, game_record: dict) -> None:
        ws         = wb.active
        ws.title   = "Conversation"
        roles      = game_record.get("roles", {})
        agent_names = game_record.get("agent_names", {})
        events     = game_record.get("events", [])

        headers = ["Day", "Phase", "Round", "Agent", "Role", "Text"]
        ws.append(headers)
        self._style_header_row(ws[1])

        # Role legend rows (before events)
        for pid in sorted(roles):
            ws.append([
                "", "Legend", "", f"P{pid}",
                _role_str(roles.get(pid)),
                agent_names.get(pid, ""),
            ])

        ws.append([])   # blank separator

        current_day = 0
        for ev in events:
            etype = ev.get("type")

            if etype == "day_start":
                current_day = ev["day"]
                row = [f"Day {current_day}", "— DAY START —", "", "", "", ""]
                ws.append(row)
                self._fill_row(ws, _FILL_DAY, _FONT_SECTION)

            elif etype == "night_start":
                row = [f"Night {ev['day']}", "— NIGHT START —", "", "", "", ""]
                ws.append(row)
                self._fill_row(ws, _FILL_NIGHT, _FONT_SECTION)

            elif etype == "talk":
                pid = ev["agent"]
                ws.append([
                    current_day, "Talk",
                    ev.get("round", 0) + 1,
                    f"P{pid}",
                    _role_str(roles.get(pid)),
                    ev.get("text", ""),
                ])

            elif etype == "vote":
                voter  = ev["agent"]
                target = ev["target"]
                ws.append([
                    current_day, "Vote", "",
                    f"P{voter}",
                    _role_str(roles.get(voter)),
                    f"→ P{target}",
                ])
                self._fill_row(ws, _FILL_VOTE)

            elif etype == "execution":
                pid = ev["agent"]
                ws.append([
                    current_day, "Execution", "",
                    f"P{pid}",
                    _role_str(roles.get(pid)),
                    f"ELIMINATED (was {ev.get('role', '?')})",
                ])
                self._fill_row(ws, _FILL_EXECUTION, _FONT_SECTION)

            elif etype == "divine":
                seer = ev["seer"]
                ws.append([
                    current_day, "Divine", "",
                    f"P{seer}",
                    _role_str(roles.get(seer)),
                    f"DIVINED P{ev['target']} → {ev['result']}",
                ])

            elif etype == "attack":
                attacks = ev.get("votes", {})
                text = "ATTACK: " + ", ".join(
                    f"P{k}→P{v}" for k, v in attacks.items()
                )
                ws.append([current_day, "Wolf Attack", "", "Wolves", "", text])

            elif etype == "guard":
                bg = ev["bodyguard"]
                ws.append([
                    current_day, "Guard", "",
                    f"P{bg}",
                    _role_str(roles.get(bg)),
                    f"Protected P{ev['target']}",
                ])

            elif etype == "guard_saved":
                ws.append([
                    current_day, "Guard Saved", "", "", "",
                    f"Attack on P{ev['target']} blocked",
                ])

            elif etype == "death":
                pid = ev["agent"]
                ws.append([
                    current_day, "Death", "",
                    f"P{pid}",
                    _role_str(roles.get(pid)),
                    f"KILLED overnight (was {ev.get('role', '?')})",
                ])
                self._fill_row(ws, _FILL_EXECUTION)

            elif etype == "game_end":
                winner = ev["winner"]
                n_days = ev.get("n_days", "?")
                winner_str = winner.value if hasattr(winner, "value") else str(winner)
                ws.append([
                    "END", "Result", "", "", "",
                    f"{winner_str} WINS in {n_days} days",
                ])
                self._fill_row(ws, _FILL_RESULT, _FONT_SECTION)

        self._auto_width(ws)

    # ------------------------------------------------------------------ #
    #  Sheet 2 — Suspicion Trace                                          #
    # ------------------------------------------------------------------ #

    def _write_suspicion_trace(
        self,
        wb:          openpyxl.Workbook,
        agent_data:  dict,       # {agent_id, agent_name, trace_snapshots, decision_weights}
        true_roles:  dict,
    ) -> None:
        agent_id        = agent_data["agent_id"]
        agent_name      = agent_data["agent_name"]
        trace_snapshots = agent_data["trace_snapshots"]
        decision_weights = agent_data["decision_weights"]
        ws = wb.create_sheet(f"Trace P{agent_id}")
        w1 = decision_weights.get("w1", 0.6)
        w2 = decision_weights.get("w2", 0.4)

        headers = [
            "Day", "Round",
            "Observing Agent", "Target", "Target Role",
            "Belief(Wolf)", "Sigma",
            "D1 Mismatch", "D2 Bandwagon", "D3 Pressure",
            "D4 Protection", "D5 Claim", "E Composite",
            f"Combined ({w1}B+{w2}S)", "Decision",
        ]
        ws.append(headers)
        self._style_header_row(ws[1])

        for snap in trace_snapshots:
            day      = snap.get("day")
            rnd      = snap.get("round")
            decision = snap.get("decision")
            beliefs  = snap.get("bayesian_beliefs", {})
            sigmas   = snap.get("sigma", {})
            detectors = snap.get("detectors", {})
            combined  = snap.get("combined_scores", {})

            for pid in sorted(detectors.keys()):
                dv       = detectors[pid]
                role_str = _role_str(true_roles.get(pid))
                is_dec   = (pid == decision)
                ws.append([
                    day, rnd,
                    f"P{agent_id} ({agent_name})",
                    f"P{pid}", role_str,
                    round(beliefs.get(pid, 0.0), 4),
                    round(sigmas.get(pid, 0.5), 4),
                    round(dv.get("d1", 0.0), 4),
                    round(dv.get("d2", 0.0), 4),
                    round(dv.get("d3", 0.0), 4),
                    round(dv.get("d4", 0.0), 4),
                    round(dv.get("d5", 0.0), 4),
                    round(dv.get("E",  0.0), 4),
                    round(combined.get(pid, 0.0), 4),
                    "VOTED" if is_dec else "",
                ])
                if is_dec:
                    self._fill_row(ws, _FILL_DECISION)

        self._auto_width(ws)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _style_header_row(row) -> None:
        for cell in row:
            cell.font      = _FONT_HEADER
            cell.fill      = _FILL_HEADER
            cell.alignment = Alignment(horizontal="center")

    @staticmethod
    def _fill_row(ws, fill, font=None) -> None:
        last_row = ws.max_row
        for cell in ws[last_row]:
            cell.fill = fill
            if font:
                cell.font = font

    @staticmethod
    def _auto_width(ws) -> None:
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max(
                (len(str(cell.value)) for cell in col if cell.value is not None),
                default=8,
            )
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)
