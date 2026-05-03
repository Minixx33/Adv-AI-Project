"""
suspicion/online_learner.py

Online gradient-descent calibration of suspicion weights across games.

After each game the evolution_rows (one row per snapshot-player pair) are used
to compute a gradient step that nudges:

  a1..a5  (detector weights in E = sum(ai*Di))
           towards making sigma(wolf) high and sigma(non-wolf) low.

  w1/w2   (combined-score decision weights: score = w1*belief + w2*sigma)
           towards making combined_score(wolf) high and combined_score(non-wolf) low.

Loss (per row): MSE
  L_sigma    = 0.5 * (sigma - y)^2   -- drives a1..a5
  L_combined = 0.5 * (combined - y)^2 -- drives w1/w2

Gradients:
  dL_sigma / d(ai)  = (sigma - y) * Di          (chain rule through E -> sigma)
  dL_combined / dw1 = (combined - y) * (belief - sigma)   (since w2 = 1 - w1)

Constraints enforced after every update:
  * Each ai clipped to [MIN_WEIGHT, 1.0] then renormalised so sum(a1..a5) = 1.
  * w1 clipped to [MIN_W, MAX_W]; w2 = 1 - w1.
"""

import csv
import os

_WOLF_ROLES = {"WEREWOLF", "POSSESSED"}
_MIN_WEIGHT = 0.01
_MIN_W      = 0.05
_MAX_W      = 0.95

_DEFAULT_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
    "w1": 0.60, "w2": 0.40,
}

_HISTORY_FIELDS = [
    "game_id", "n_rows",
    "a1", "a2", "a3", "a4", "a5",
    "w1", "w2",
    "avg_loss_sigma", "avg_loss_combined",
]


class OnlineLearner:
    """
    Maintains the live suspicion weights and updates them after every game.

    Parameters
    ----------
    initial_weights : dict | None
        Starting a1..a5 / w1 / w2.  Defaults to the module defaults.
    lr : float
        Learning rate applied to each gradient step (default 0.01).
    """

    def __init__(self, initial_weights: dict = None, lr: float = 0.01):
        self._weights = dict(_DEFAULT_WEIGHTS)
        if initial_weights:
            self._weights.update(initial_weights)
        self._lr      = lr
        self._history: list = []   # one dict per game, appended by update()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def update(self, game_id: int, evolution_rows: list) -> dict:
        """
        Consume evolution_rows from one game and return the updated weights.
        Appends one row to self.history.
        evolution_rows may come from multiple suspicion agents in the same game.
        """
        if not evolution_rows:
            return dict(self._weights)

        # ── Accumulate gradients ──────────────────────────────────────
        grad_a = {k: 0.0 for k in ("a1", "a2", "a3", "a4", "a5")}
        grad_w1 = 0.0
        sum_loss_sigma    = 0.0
        sum_loss_combined = 0.0
        n = 0

        for row in evolution_rows:
            role = (row.get("target_role") or "VILLAGER").upper()
            y    = 1.0 if role in _WOLF_ROLES else 0.0

            sigma    = float(row.get("sigma",          0.5))
            belief   = float(row.get("belief_wolf",    0.5))
            combined = float(row.get("combined_score", 0.5))

            err_sigma    = sigma    - y
            err_combined = combined - y

            sum_loss_sigma    += 0.5 * err_sigma    ** 2
            sum_loss_combined += 0.5 * err_combined ** 2

            for i, key in enumerate(("a1", "a2", "a3", "a4", "a5"), 1):
                d = float(row.get(f"d{i}", 0.0))
                grad_a[key] += err_sigma * d

            # w1 gradient: dL/dw1 = err_combined * (belief - sigma)
            grad_w1 += err_combined * (belief - sigma)

            n += 1

        # ── Average gradients ─────────────────────────────────────────
        for k in grad_a:
            grad_a[k] /= n
        grad_w1 /= n

        # ── Update a1..a5 ─────────────────────────────────────────────
        new_a = {}
        for k in ("a1", "a2", "a3", "a4", "a5"):
            new_a[k] = max(_MIN_WEIGHT, self._weights[k] - self._lr * grad_a[k])
        # Renormalise to sum to 1
        total = sum(new_a.values())
        for k in new_a:
            new_a[k] = new_a[k] / total

        # ── Update w1 (w2 = 1 - w1) ───────────────────────────────────
        new_w1 = self._weights["w1"] - self._lr * grad_w1
        new_w1 = max(_MIN_W, min(_MAX_W, new_w1))
        new_w2 = 1.0 - new_w1

        # ── Commit ────────────────────────────────────────────────────
        self._weights.update(new_a)
        self._weights["w1"] = round(new_w1, 6)
        self._weights["w2"] = round(new_w2, 6)

        self._history.append({
            "game_id": game_id,
            "n_rows":  n,
            "a1": round(new_a["a1"], 6),
            "a2": round(new_a["a2"], 6),
            "a3": round(new_a["a3"], 6),
            "a4": round(new_a["a4"], 6),
            "a5": round(new_a["a5"], 6),
            "w1": round(new_w1, 6),
            "w2": round(new_w2, 6),
            "avg_loss_sigma":    round(sum_loss_sigma    / n, 6),
            "avg_loss_combined": round(sum_loss_combined / n, 6),
        })

        return dict(self._weights)

    def get_weights(self) -> dict:
        return dict(self._weights)

    @property
    def history(self) -> list:
        return list(self._history)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def write_history(self, output_dir: str) -> str:
        """Write weight_history.csv to output_dir. Returns the file path."""
        path = os.path.join(output_dir, "weight_history.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._history)
        return path
