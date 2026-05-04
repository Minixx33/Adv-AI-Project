# AIWolfPy

Create python agents that can play Werewolf, following the specifications of the [AIWolf Project](http://aiwolf.org)

This has been forked from the official repository by the AIWolf project, and was originally created by [Kei Harada](https://github.com/k-harada).

# Changelog:

## Version 0.4.9a
* Added support material in English

## Version 0.4.9
* Changed differential structure (diff_data) into a DataFrame

## Version 0.4.4
* removed daily_finish
* Added update callback (with request parameter)
* Connecting is now done through a instance, not a class

## Version 0.4.0
* Support for python3
* Made file structure much simpler

# Running the agent and the server locally:
* Download the AIWolf platform from the [AIWolf public website] (http://www.aiwolf.org/server/)
	* Don't forget that the local AIWolf server requires JDK 11
* Start the server with `./StartServer.sh`
	* This runs a Java application. Select the number of players, the connection port, and press "Connect".
* In another terminal, run the client management application `./StartGUIClient.sh`
	* Another Java application is started. Select the client jar file (sampleclient.jar), the sample client pass, and the port configured for the server.
	* Press "Connect" for each instance of the sample agent you wish to connect.
* Run the python agent from this repository, with the command: `./python_sample.py -h [hostname] -p [port]`
* On the server application, press "Start Game".
  * The server application will print the log to the terminal, and also to the application window. Also, a log file will be saved on "./log".
* You can see a fun visualization using the "log viewer" program.

# Running the Suspicion-Aware Werewolf Simulation (Model_V2)

This repository includes a self-contained Python simulation engine (`Model_V2/`) that does **not** require the Java AIWolf server. All agents, the suspicion module, and experiment runners are pure Python.

## Dependencies

```bash
pip install scipy numpy pandas openpyxl
```

## Run a single game

```bash
python Model_V2/run_game.py --config Model_V2/configs/default.json
```

## Run a batch experiment (N games)

```bash
python Model_V2/run_game.py \
    --config   Model_V2/configs/default.json \
    --n-games  10000 \
    --seed     42 \
    --output   results/baseline/seed_42/
```

Repeat with `--seed 1`, `--seed 2`, `--seed 3` (or whichever seeds you use) for each experimental condition.

## Run the two-stage hyperparameter grid search

```bash
python -c "
from Model_V2.runner.grid_search import GridSearch
import json
config = json.load(open('Model_V2/configs/default.json'))
gs = GridSearch(config, n_games=2000, n_samples=200, seed=0)
gs.run('results/grid_search/')
"
```

This runs Stage 1 (Latin Hypercube Sampling over detector weights a1–a5, ~200 configs) and Stage 2 (sweep w1 from 0.0 to 1.0, 11 configs). Results are saved to `results/grid_search/grid_search_results.csv`.

## Post-hoc statistical analysis

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
- **Mann-Whitney U test** on continuous metrics: sigma gap, wolf vote precision/recall, false accusation rate, per-detector means D1–D5 (effect size: rank-biserial r)

Comparisons run automatically: baseline vs all conditions, plus the mixed-population dose-response chain. Significance stars (`***` / `**` / `*` / `ns`) and a seed-level stability table are printed to stdout. Use `--alpha` to change the significance threshold (default: 0.05).

## Outputs per run

Each game batch produces the following files in its output directory:

| File | Contents |
|---|---|
| `conversation.txt` | Full talk log for every game |
| `suspicion_trace.txt` | Per-round σ scores and detector values with token evidence |
| `suspicion_evolution.csv` | Time-series suspicion data for plotting |
| `game_summary.json` | Per-game structured results |
| `summary.csv` | Aggregate metrics across all games (input to `stats_analysis.py`) |
| `*.xlsx` | Excel summary exported by XlsxExporter |

---

# Running the agent on the AIWolf competition server:
* After you create your account in the competition server, make sure your client's name is the same as your account's name.
* The python packages available at the competition server are listed in this [page](http://aiwolf.org/python_modules)
* You can expect that the usual packages + numpy, scipy, pandas, scikit-learn are available.
	* Make sure to check early with the competition runners, specially if you want to use something like an specific version of tensorflow.
	* The competition rules forbid running multiple threads. Numpy and Chainer are correctly set-up server side, but for tensorflow you must make sure that your program follows this rule. Please see the following [post](http://aiwolf.org/archives/1951)
* For more information, a tutorial from the original author of this package can be seen in this [slideshare](https://www.slideshare.net/HaradaKei/aiwolfpy-v049) (in Japanese).
