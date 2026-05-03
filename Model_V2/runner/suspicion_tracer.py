"""
runner/suspicion_tracer.py

Formats the suspicion-agent reasoning trace into the human-readable file
specified in §3.5 of the architecture.

Hierarchy / contents:
  SuspicionTracer
    └── write(game_record, trace_snapshots, output_dir)
          Writes <output_dir>/game_<id>/suspicion_trace.txt

Output format (spec §3.5):
  === SUSPICION AGENT TRACE: P3 (BayesianAgent+Suspicion) ===
  True role: Villager

  --- DAY 1, END OF ROUND 5 ---

  Bayesian beliefs P(p = wolf):
    P1: 0.20    P2: 0.45    ...

  Suspicion scores σ(p):
    P1: 0.50    P2: 0.65    ...

  Detector activations this round:
    P2:
      D1 (vote-accusation mismatch): 0.50  → P2 accused P3 but voted P5
      ...
      Composite E(P2): 0.19           → suspicion DECREASED slightly

  Combined decision scores: score(p) = w1*belief + w2*σ
    P1: 0.32    ...

  DECISION: Vote P8 (highest combined score)
  RATIONALE: ...

  --- DAY 2, END OF ROUND 5 ---
  ...
"""

import os
from io import StringIO


class SuspicionTracer:
    """Writes suspicion_trace.txt from trace snapshots."""

    def write(
        self,
        game_record:      dict,
        trace_snapshots:  list,
        agent_id:         int,
        agent_name:       str,
        true_roles:       dict,
        decision_weights: dict,
        output_dir:       str,
    ) -> str:
        """
        Render and save the suspicion trace.
        Returns the path to the written file.
        """
        gid    = game_record["game_id"]
        folder = os.path.join(output_dir, f"game_{gid:04d}")
        os.makedirs(folder, exist_ok=True)
        path   = os.path.join(folder, "suspicion_trace.txt")

        text = self._render(
            game_record, trace_snapshots, agent_id,
            agent_name, true_roles, decision_weights
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render(
        self,
        rec:              dict,
        snapshots:        list,
        agent_id:         int,
        agent_name:       str,
        true_roles:       dict,
        decision_weights: dict,
    ) -> str:
        buf = StringIO()
        w1  = decision_weights.get("w1", 0.6)
        w2  = decision_weights.get("w2", 0.4)

        agent_role = rec["roles"].get(agent_id, "Unknown")
        buf.write(
            f"=== SUSPICION AGENT TRACE: P{agent_id} ({agent_name}) ===\n"
        )
        buf.write(f"True role: {agent_role}\n\n")

        for snap in snapshots:
            day = snap.get("day", "?")
            rnd = snap.get("round", "?")
            buf.write(f"--- DAY {day}, END OF ROUND {rnd} ---\n\n")

            # Bayesian beliefs
            beliefs = snap.get("bayesian_beliefs", {})
            buf.write("Bayesian beliefs P(p = wolf):\n")
            buf.write(self._format_player_row(beliefs))
            buf.write("\n")

            # Suspicion scores
            sigmas = snap.get("sigma", {})
            buf.write("Suspicion scores σ(p):\n")
            buf.write(self._format_player_row(sigmas))
            buf.write("\n")

            # Detector activations
            detectors = snap.get("detectors", {})
            evidence  = snap.get("evidence", {})
            buf.write("Detector activations this round:\n")
            for pid in sorted(detectors.keys()):
                dv        = detectors[pid]
                sigma_now = sigmas.get(pid, 0.5)
                # Threshold 0.15 matches the update rule in SuspicionModule
                direction = "INCREASED" if dv["E"] > 0.15 else "DECREASED"
                ev        = evidence.get(pid, {})
                buf.write(f"  P{pid}:\n")

                buf.write(f"    D1 (vote-accusation mismatch): {dv['d1']:.2f}\n")
                for _, desc in (ev.get("d1") or [])[-2:]:
                    buf.write(f"      Evidence: {desc}\n")

                buf.write(f"    D2 (bandwagon):                {dv['d2']:.2f}\n")
                for _, desc in (ev.get("d2") or [])[-2:]:
                    buf.write(f"      Evidence: {desc}\n")

                buf.write(f"    D3 (pressure-trigger):         {dv['d3']:.2f}\n")
                # Only show D3 evidence when the detector is non-zero;
                # zero-pressure entries are informative only when D3 fired.
                if dv["d3"] > 0:
                    for _, desc in (ev.get("d3") or [])[-2:]:
                        buf.write(f"      Evidence: {desc}\n")

                buf.write(f"    D4 (protection):               {dv['d4']:.2f}\n")
                for _, token_text, defended in (ev.get("d4") or [])[-2:]:
                    buf.write(f"      Evidence: P{pid} emitted \"{token_text}\" (defending P{defended})\n")

                buf.write(f"    D5 (claim consistency):        {dv['d5']:.2f}\n")
                for _, desc in (ev.get("d5") or [])[-2:]:
                    buf.write(f"      Evidence: {desc}\n")

                buf.write(
                    f"    Composite E(P{pid}): {dv['E']:.2f}"
                    f"           -> suspicion {direction} to {sigma_now:.2f}\n\n"
                )

            # Combined scores
            combined = snap.get("combined_scores", {})
            buf.write(
                f"Combined decision scores: score(p) = {w1}*belief + {w2}*σ\n"
            )
            buf.write(self._format_player_row(combined))
            buf.write("\n")

            # Decision
            decision = snap.get("decision")
            buf.write(f"DECISION: Vote P{decision} (highest combined score)\n")

            # Rationale
            if decision in detectors and decision in beliefs:
                dv     = detectors[decision]
                belief = beliefs.get(decision, 0.0)
                sigma  = sigmas.get(decision, 0.5)
                buf.write("RATIONALE:\n")
                buf.write(f"  - Highest belief that P{decision} is wolf: {belief:.2f}\n")
                buf.write(f"  - Highest suspicion: {sigma:.2f}\n")
                if dv["d5"] > 0.5:
                    buf.write(
                        f"  - D5 claim contradiction is the primary driver "
                        f"(score {dv['d5']:.2f})\n"
                    )
                if dv["d1"] > 0.4:
                    buf.write(
                        f"  - D1 vote-accusation mismatch is significant "
                        f"({dv['d1']:.2f})\n"
                    )
            buf.write("\n")

        return buf.getvalue()

    @staticmethod
    def _format_player_row(pid_values: dict) -> str:
        """Format {pid: value} as compact rows of four players each."""
        buf = StringIO()
        items = sorted(pid_values.items())
        for start in range(0, len(items), 4):
            chunk = items[start:start + 4]
            buf.write("  " + "    ".join(f"P{p}: {v:.2f}" for p, v in chunk))
            buf.write("\n")
        return buf.getvalue()
