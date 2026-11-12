# Task 2 plan — interactive Dash dashboard and deployment

## Context

- Dataset: Adult Census Income, raw file `datasets/census_income.csv`.
- Existing Task 1 code: `part_1/scripts/prepare_dataset.py`, `part_1/src/task1/`, `part_1/run.py`.
- Existing Task 1 artefacts:
  - `part_1/data/nn_pred.csv` — `real` (0/1) and `pred` (continuous regressor score).
  - `part_1/data/rf_pred.csv` — `real` and `pred` (hard boolean labels only).
  - `part_1/data/nn_roc.csv`, `part_1/data/nn_prc.csv`.
  - `part_1/plots/*.png`.
- Task 1 report: `reports/task1.md`. Useful reference values: 48,598 cleaned rows, 23.55% positive class, NN PR-optimal threshold 0.362, ROC-optimal threshold 0.250.

## Decisions taken in this plan

| Topic | Decision | Reason |
| --- | --- | --- |
| Cleaned dataset | Generate `part_1/dataset/census_clean.csv` from the raw CSV | Keeps raw data untouched and gives Task 2 its own reproducible input |
| Categorical features | Keep them as strings; never one-hot encode them in the saved dataset | Explicit Task 2 requirement; keeps the table and charts readable |
| Missing markers | Replace `?` with `Unknown`; do not drop those rows | Preserves 5–6% of rows and makes the marker explicit |
| Sentinel values | Remove rows where `capital-gain == 99999`; check `capital-loss` for the same value | Task 1 identified 99999 as a sentinel, not a real gain |
| Dropped columns | `fnlwgt`, `education-num` | Task 1 decision; keep the cleaned dataset aligned with the analysis |
| Target | Keep normalised `income` string and add boolean `target` | Readable in the table; easy to use for counts and metrics |
| App data source | Read saved CSV artefacts at startup; never retrain models at runtime | Fast, deterministic, and simple to containerise |
| App layout | Two tabs in one Dash app: **Data explorer** and **Model evaluation** | Meets the “another page” requirement without extra routing complexity |
| RF comparison | Add RF probabilities to Task 1 artefacts (recommended); fall back to hard labels if skipped | A shared threshold is much more useful for comparing the two models |
| Deployment | Dockerfile + Docker Compose as the minimum; Kubernetes manifests for minikube as the advanced path | Matches the Task 2 guide and the timebox |

## How to use this file

Run the prompts in order, one at a time. After each prompt:

1. Read the agent’s diff.
2. Run the acceptance checks.
3. Commit with a short conventional-commit message.
4. Add one line to `part_1/AI_LOG.md` (time, purpose, what was kept).
5. Only then start the next prompt.

If a prompt fails, fix it before continuing. Do not let an agent start the next prompt on top of a broken state.

## Global constraints for every prompt

- Work inside `part_1/` unless the prompt explicitly mentions `reports/`, the repository root `.dockerignore`, or `datasets/`.
- Never modify `datasets/census_income.csv`.
- Never one-hot encode categorical features in the saved dataset. Do not use `pd.get_dummies` or `OneHotEncoder` for `census_clean.csv`.
- Do not retrain the Task 1 models when the dashboard starts. Read the saved CSVs.
- Keep the existing Task 1 pipeline working. Only the RF-probability prompt may touch `part_1/src/task1/`, and it must keep `rf_pred.csv` backwards compatible.
- Use `uv run` from `part_1/` for Python commands.
- Prefer pure functions for cleaning, metrics, and correlation so they can be tested without starting Dash.
- Every chart must have a title and axis labels. Add a one-sentence caption or note that states the claim.
- Do not commit generated cleaned data or large artefacts. Add generated files to `.gitignore` if appropriate; the app must regenerate or read them through a documented command.
- No secrets or API keys. Configuration goes through environment variables.
- Commit after each prompt. Use messages such as `feat(task2): ...`, `test(task2): ...`, `docs(task2): ...`, `build(task2): ...`.

## Expected file layout

```text
part_1/
  dataset/
    census_clean.csv              # generated, not committed
    census_clean_meta.json        # generated, not committed
  scripts/
    clean_dataset.py              # CLI wrapper around cleaning.py
  src/task2/
    __init__.py
    cleaning.py                   # pure cleaning functions
    data.py                       # load cleaned data and predictions
    metrics.py                    # confusion-matrix metrics and correlation helpers
    app.py                        # create_app(), layout, callbacks, server
  run_dashboard.py                # local entry point
  tests/
    test_task2.py                 # cleaning, metrics, correlation, app smoke tests
  Dockerfile
  .dockerignore                   # at repository root if build context is the root
  docker-compose.yml
  k8s/                            # optional minikube manifests
    configmap.yaml
    deployment.yaml
    service.yaml
```

---

## Prompt 0 — Baseline audit (read-only)

````markdown
You are working in the repository `/home/work/PycharmProjects/ai_mlops_hands_on_exercise`.

This is a read-only audit. Do not modify any file.

Inspect:
- `datasets/census_income.csv`: header, leading spaces in column names, unique values of `income`, occurrences of `?`, and occurrences of `99999`.
- `part_1/scripts/prepare_dataset.py`, `part_1/src/task1/*.py`, `part_1/run.py`, `part_1/pyproject.toml`, `part_1/uv.lock`.
- `part_1/data/nn_pred.csv`, `part_1/data/rf_pred.csv`, `part_1/data/nn_roc.csv`, `part_1/data/nn_prc.csv`.
- `reports/task1.md`.

Report, with exact numbers:
1. Raw row count and column names after stripping whitespace.
2. Target counts and positive rate before and after removing `capital-gain == 99999`.
3. Counts of `?` per categorical column.
4. Columns that must remain after dropping `fnlwgt` and `education-num`.
5. Numeric and categorical column lists for the cleaned dataset.
6. Minimum, maximum, and default candidates for the NN score threshold.
7. Whether `rf_pred.csv` contains probabilities or only hard labels.
8. Any blocker for building a Dash app from the saved artefacts.

Do not create files. Stop after the report and wait for the next prompt.
````

**Acceptance:** a report with the numbers above; no repository changes.

---

## Prompt 1 — Clean the dataset without one-hot encoding

````markdown
Implement the Task 2 cleaning step.

Files to create:
- `part_1/src/task2/cleaning.py`
- `part_1/scripts/clean_dataset.py`
- `part_1/tests/test_task2.py` (cleaning tests for now)

Requirements for `part_1/src/task2/cleaning.py`:
1. Provide `clean_census(df: pandas.DataFrame) -> pandas.DataFrame`.
2. Strip leading/trailing whitespace from every column name.
3. Remove rows where `capital-gain == 99999`. Scan `capital-loss` for the same sentinel and remove those rows too if present. Print or return the number of removed rows.
4. Drop `fnlwgt` and `education-num`.
5. Normalise `income`:
   - strip whitespace and trailing periods;
   - keep only `<=50K` and `>50K`;
   - keep the normalised `income` string column;
   - add a boolean `target` column (`True` for `>50K`).
6. Replace `?` with `Unknown` in `workclass`, `occupation`, and `native-country`.
7. Do **not** one-hot encode anything. Categorical columns must remain string/object columns.
8. Preserve the original row order otherwise. Do not deduplicate rows.
9. Return a copy; do not mutate the caller’s DataFrame.

Expected cleaned columns, in a sensible order:
`age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, `income`, `target`.

Requirements for `part_1/scripts/clean_dataset.py`:
- Resolve paths from `__file__`, not from the current working directory.
- Read `datasets/census_income.csv` from the repository root.
- Write `part_1/dataset/census_clean.csv` with `index=False`.
- Write `part_1/dataset/census_clean_meta.json` containing:
  - source path;
  - rows before and after cleaning;
  - rows removed for sentinel values;
  - positive count and positive rate;
  - cleaned columns, numeric columns, categorical columns;
  - counts of `?` replaced per column.
- Print a short summary: rows before, rows removed, rows after, positive rate, output path.
- Do not save the one-hot `part_1/dataset/census.csv`; Task 1 owns that file.

Tests in `part_1/tests/test_task2.py`:
- Use the standard-library `unittest` module.
- Build a tiny DataFrame that includes a 99999 sentinel, a `?`, an `income` value with a trailing period, `fnlwgt`, and `education-num`.
- Assert the sentinel row is removed, both columns are dropped, `income` is normalised, `target` is boolean, `?` becomes `Unknown`, and no one-hot columns appear.
- Add an integration test that reads the real raw CSV, runs the cleaning function, and asserts:
  - 48,598 rows (or report the exact difference if the raw file changed);
  - 11,443 positives and a positive rate near 23.55%;
  - `capital-gain.max() < 99999`;
  - `fnlwgt` and `education-num` are absent;
  - the cleaned columns match the expected list exactly.

Acceptance checks:
```bash
cd part_1
uv run python scripts/clean_dataset.py
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Also verify with a small Python snippet that:
- the saved CSV has no columns starting with `workclass_`, `education_`, `occupation_`, etc.;
- `target.dtype` is boolean;
- the categorical columns are not numeric.

Do not modify `part_1/src/task1/` or `datasets/`.
Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add cleaned census dataset`.
````

**Acceptance:** the cleaning script runs, the tests pass, the cleaned CSV exists locally, and the saved dataset has no one-hot columns.

---

## Prompt 2 — Add RF probabilities to the Task 1 artefacts (recommended)

> Skip this prompt only if you are sure you want the RF comparison fixed at its hard 0.5 threshold. If you skip it, the model-evaluation prompt must clearly label the RF result as “hard predictions at 0.5”.

````markdown
Make the random forest comparable at any threshold, without changing the existing Task 1 outputs in a breaking way.

Changes:
1. In `part_1/src/task1/models.py`, make `random_forest_predict` return both:
   - the hard predictions from `RandomForestClassifier.predict(X_test)`;
   - the positive-class probabilities from `RandomForestClassifier.predict_proba(X_test)[:, 1]`.
   You may return a tuple or a small dataclass. Keep the function name clear.
2. In `part_1/src/task1/main.py`, save `part_1/data/rf_pred.csv` with columns:
   - `real`: unchanged;
   - `pred`: unchanged hard boolean labels;
   - `proba`: the new continuous probability.
   Keep `nn_pred.csv` unchanged.
3. Keep `part_1/src/task1/plot.py` and `part_1_plot` working with the new `rf_pred.csv`. They must still read the `pred` column.
4. Re-run the Task 1 pipeline so the artefacts are regenerated:
```bash
cd part_1
uv run python scripts/prepare_dataset.py
uv run python run.py
```
5. Verify:
   - `rf_pred.csv` has `real`, `pred`, `proba`;
   - `proba` is in `[0, 1]`;
   - at a 0.5 threshold, `proba > 0.5` matches the hard `pred` except possibly for ties;
   - the five Task 1 plots still regenerate;
   - the Task 1 report metrics still hold.
6. Do not edit `reports/task1.md`. Mention the new `proba` column in the Task 2 report later.
7. If you cannot make this change safely, stop and report the reason. Do not silently retrain a second model just to obtain probabilities.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): save random forest probabilities for thresholding`.
````

**Acceptance:** `rf_pred.csv` contains a continuous `proba` column, the old `pred` column still behaves as before, and the Task 1 plots still regenerate.

---

## Prompt 3 — Data layer and Dash app skeleton

````markdown
Create the data-loading layer and the empty two-tab Dash application.

Files to create or update:
- `part_1/src/task2/data.py`
- `part_1/src/task2/app.py`
- `part_1/src/task2/__init__.py`
- `part_1/run_dashboard.py`

`part_1/src/task2/data.py`:
1. Define defaults resolved from the package location, not the current working directory:
   - cleaned dataset: `part_1/dataset/census_clean.csv`;
   - predictions directory: `part_1/data`.
2. Allow overrides through environment variables:
   - `CENSUS_DATA_PATH`;
   - `PREDICTIONS_DIR`.
3. Provide `load_clean_dataset(path: str | Path | None = None) -> pandas.DataFrame`:
   - read the CSV;
   - validate the required columns;
   - cast `target` to boolean and `income` to string;
   - return a copy.
   - If the file is missing, raise a `FileNotFoundError` with this exact actionable message: `Cleaned dataset not found. Run: cd part_1 && uv run python scripts/clean_dataset.py`.
4. Provide `load_predictions(pred_dir: str | Path | None = None)` returning a small dataclass or dictionary with:
   - `y_true`: integer 0/1 labels from `nn_pred.csv` (and verify `rf_pred.csv` has the same labels in the same row order);
   - `nn_score`: continuous NN score;
   - `rf_score`: `rf_pred.csv["proba"]` if present, otherwise the hard `pred` cast to 0/1;
   - `rf_has_proba`: boolean;
   - `threshold_default`: `0.362` (Task 1 PR-optimal) if it lies inside the combined score range, otherwise `0.5`.
   - Read the CSVs with `index_col=0`.
5. Provide helpers `numeric_columns(df)` and `categorical_columns(df)` that use pandas dtypes and do not treat `target` as a categorical feature.

`part_1/src/task2/app.py`:
1. Provide `create_app(data_path=None, pred_dir=None) -> Dash`.
2. Load the data once inside `create_app` and capture it in the callbacks. Do not reload CSVs on every callback.
3. Use `dash_bootstrap_components` with the `BOOTSTRAP` theme.
4. Layout:
   - a compact header with the title “Adult Census Income Explorer” and a subtitle;
   - `dbc.Tabs` with two tabs: `Data explorer` and `Model evaluation`;
   - each tab currently contains a placeholder card;
   - a footer that names the dataset source and says the models are evaluated on the Task 1 held-out test set.
5. Add `@app.server.route("/health")` returning `{"status": "ok"}` so the container healthcheck can use it.
6. Expose:
   - `app = create_app()`;
   - `server = app.server`.
7. Provide `main()` that reads `HOST` (default `0.0.0.0`), `PORT` (default `8050`), and `DASH_DEBUG` (default `false`), then calls `app.run`.
8. Do not suppress callback exceptions unless you actually need dynamic component IDs.

`part_1/run_dashboard.py`:
- Import the `task2` package the same way `part_1/run.py` imports `task1`.
- Call the app entry point.

Acceptance checks:
```bash
cd part_1
uv run python scripts/clean_dataset.py
uv run python run_dashboard.py
```
- The app starts at `http://127.0.0.1:8050`.
- Both tabs render without a browser console error.
- `/health` returns `{"status": "ok"}`.
- No model training happens at startup.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add dashboard data layer and app shell`.
````

**Acceptance:** the app starts, both tabs are visible, `/health` works, and the data is loaded once.

---

## Prompt 4 — Data explorer: table

````markdown
Implement the table view in the Data explorer tab.

Controls:
- `viz-type`: a `dcc.Dropdown` with `Table`, `Histogram`, `Correlation`; default `Table`.
- `feature-select`: a multi-select `dcc.Dropdown` containing every cleaned column; default all columns.
- `select-all` and `clear-features` buttons.
- A small summary line: number of rows, number of columns, and positive-class rate.

Output:
- A single `html.Div(id="explorer-output")` inside a `dcc.Loading`.
- When `viz-type == "Table"`, render a `dash_table.DataTable` with:
  - only the selected columns; if nothing is selected, use all columns;
  - `page_size=15`, `page_action="native"`, `sort_action="native"`, `filter_action="native"`, `fixed_rows={"headers": True}`;
  - `export_format="csv"` if it works with the installed Dash version;
  - readable column headers (replace `-` with spaces, title-case);
  - `target` displayed as `>50K` / `<=50K` rather than `True` / `False`;
  - a compact style with striped rows and a sticky header.
- Show a `dbc.Alert` if the selection is empty after filtering.

Implementation notes:
- Do not mutate the loaded DataFrame when formatting the target. Build a display copy.
- The table must show the cleaned dataset, **not** the one-hot `part_1/dataset/census.csv`.
- Keep the callback pure: one callback input (`viz-type`, `feature-select`) and one output (`explorer-output`). Add the select-all/clear buttons through a separate small callback that only changes `feature-select`.
- Do not add histogram or correlation logic yet; return a placeholder for those options.

Acceptance checks:
- The table shows 48,598 rows.
- Selecting a few features changes the visible columns.
- Sorting and filtering work.
- The target column displays labels, not booleans.
- No one-hot column names appear anywhere.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add interactive dataset table`.
````

**Acceptance:** the cleaned table is browsable, sortable, filterable, and column-selectable.

---

## Prompt 5 — Data explorer: histograms

````markdown
Add the histogram view to the Data explorer tab.

Behaviour:
- When `viz-type == "Histogram"`, use the selected features.
- Allow 1–4 selected features. If more than 4 are selected, show a `dbc.Alert` saying “Select at most four features to compare distributions”.
- If no feature is selected, show an alert.
- For each selected numeric feature, draw a histogram with 30 bins, counts on the y-axis, and the feature name on the x-axis.
- For each selected categorical feature, draw a bar chart of value counts, sorted from most to least frequent.
- Use `plotly.subplots.make_subplots` so 1–4 features appear in a responsive grid (up to two columns).
- Set a readable figure height, e.g. `320 * number_of_rows`.
- Give every subplot a title and axis labels. Replace `-` in feature names with spaces in titles.
- If the same callback also handles the table, keep the table branch unchanged.
- Do not one-hot encode categorical features to draw the bars; use `value_counts()` directly.

Optional but useful: add a `split-by-target` checklist. When enabled:
- numeric features show two overlaid histograms (`<=50K` and `>50K`), with opacity around 0.6;
- categorical features show grouped bars by target.
If you add it, make sure the default is off and the legend is labelled.

Acceptance checks:
- `age` produces a histogram.
- `workclass` produces a sorted bar chart.
- Selecting `age` and `hours-per-week` produces two subplots.
- Selecting five features produces the alert, not a crash.
- No one-hot columns are created in the saved data.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add feature distribution charts`.
````

**Acceptance:** numeric and categorical distributions render for 1–4 selected features.

---

## Prompt 6 — Data explorer: correlations

````markdown
Add the correlation view to the Data explorer tab.

Controls:
- A `corr-method` radio item with two options:
  - `Numeric (Pearson)`;
  - `Categorical (Cramér's V)`.
- Reuse `feature-select`. Filter the selected columns according to the chosen method.

Numeric mode:
- Require at least two numeric features.
- Compute the Pearson correlation matrix with `df[cols].corr(method="pearson")`.
- Render a Plotly heatmap:
  - `zmin=-1`, `zmax=1`;
  - a diverging colour scale (`RdBu` or similar);
  - annotated values to two decimals;
  - title “Pearson correlation between numeric features”;
  - readable feature names on both axes.
- Optionally add a small table of the five strongest absolute off-diagonal correlations.

Categorical mode:
- Require at least two categorical features.
- Compute Cramér's V for every pair using a contingency table:
  - `pd.crosstab(df[a], df[b])`;
  - `scipy.stats.chi2_contingency`;
  - `V = sqrt(chi2 / (n * min(rows - 1, cols - 1)))`;
  - return `0.0` when the denominator is zero.
- Render a heatmap with `zmin=0`, `zmax=1`, annotated values, and the title “Cramér's V association between categorical features”.
- Do not one-hot encode the categorical features; use the contingency tables directly.
- Add `scipy` as an explicit dependency if it is not already direct:
```bash
cd part_1
uv add scipy
```

Shared behaviour:
- If fewer than two valid features are selected, show a `dbc.Alert` explaining what to select.
- Every heatmap must have a colour bar with a clear label.
- Keep the table and histogram branches unchanged.

Acceptance checks:
- Selecting two numeric features produces a symmetric matrix with a diagonal of 1.
- Selecting two categorical features produces values between 0 and 1.
- Switching the method changes the available/used features and the title.
- The saved cleaned dataset still has no one-hot columns.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add numeric and categorical correlation views`.
````

**Acceptance:** Pearson and Cramér's V heatmaps work and are driven by the user’s feature selection.

---

## Prompt 7 — Model evaluation: threshold and RF comparison

````markdown
Implement the Model evaluation tab.

Data:
- Use `load_predictions()` from `part_1/src/task2/data.py`.
- `y_true` is the held-out Task 1 test set.
- `nn_score` is the continuous `nn_pred.csv["pred"]`.
- `rf_score` is `rf_pred.csv["proba"]` when present; otherwise the hard `pred` cast to 0/1.
- Do not retrain anything.

Metrics helper in `part_1/src/task2/metrics.py`:
- `confusion_metrics(y_true, y_score, threshold) -> dict` returning:
  - `tn`, `fp`, `fn`, `tp`;
  - `accuracy`, `precision`, `recall`, `f1`;
  - `predicted_positive`.
- Use the comparison `y_score > threshold` to match the Task 1 code. Document this choice in a docstring.
- Handle zero denominators with `0.0`.

Controls:
- `threshold-slider`:
  - range from the combined minimum to the combined maximum of NN and RF scores, rounded outward to two decimals;
  - step `0.001`;
  - default `0.362` if it lies in range, otherwise `0.5`;
  - marks at the minimum, `0`, `0.25`, `0.5`, `0.75`, `1`, and the maximum, de-duplicated and sorted;
  - tooltip always visible.
- `compare-rf`: a `dbc.Switch` labelled “Compare with Random Forest”; default off.

Outputs:
- NN confusion matrix as a Plotly heatmap:
  - rows: actual `<=50K`, actual `>50K`;
  - columns: predicted `<=50K`, predicted `>50K`;
  - counts annotated;
  - title includes the current threshold.
- Metric cards for accuracy, precision, recall, and F1.
- A compact counts line for TP, FP, FN, TN, and the number of predicted positives.
- A note: “The neural network is a regressor; its scores are not calibrated probabilities.”
- When `compare-rf` is on:
  - show the RF confusion matrix next to the NN one;
  - both use the same threshold when `rf_has_proba` is true;
  - show a comparison `dash_table.DataTable` with one row per model and columns for accuracy, precision, recall, F1, TP, FP, FN, TN;
  - if `rf_has_proba` is false, render the RF matrix from hard labels and add the note “RF has no saved probabilities; this matrix is fixed at the 0.5 hard-label cut-off.”
- Use `dcc.Loading` around the output area.
- Keep the layout responsive: controls in a card, metric cards in a row, matrices side by side when comparing.

Acceptance checks:
- Moving the slider changes the NN confusion matrix and all NN metrics.
- At threshold `0.362`, the NN metrics approximately match the Task 1 report:
  - accuracy ≈ 0.841;
  - precision ≈ 0.645;
  - recall ≈ 0.738;
  - F1 ≈ 0.689.
  If the exact counts differ by one because of `>` versus `>=`, document it; do not silently change the criterion.
- With RF probabilities, the RF matrix also changes with the threshold.
- Toggling `compare-rf` shows and hides the RF matrix and comparison table.
- The app never retrains a model.

Add a unit test in `part_1/tests/test_task2.py`:
- a hand-computed 2×2 example for `confusion_metrics`;
- a check of the real NN metrics at `0.362` with tolerances of `0.001` for the four metrics.

Update `part_1/AI_LOG.md`.
Commit as `feat(task2): add thresholded model evaluation page`.
````

**Acceptance:** the threshold controls the NN confusion matrix, and the RF toggle compares both models at the same threshold.

---

## Prompt 8 — Polish, validation, and app tests

````markdown
Polish the dashboard and add regression tests.

User experience:
- Add `dcc.Loading` where callbacks can take a moment.
- Add friendly alerts for:
  - missing cleaned dataset;
  - missing prediction files;
  - empty feature selection;
  - too many histogram features;
  - fewer than two features for correlation.
- Add a short “About the data” note in the Data explorer tab:
  - 48,598 rows after cleaning;
  - 244 sentinel rows removed;
  - `fnlwgt` and `education-num` dropped;
  - categorical features kept as strings, not one-hot encoded;
  - `?` replaced by `Unknown`.
- Add a short “About the models” note in the Model evaluation tab:
  - metrics are computed on the Task 1 80/20 holdout;
  - NN is an `MLPRegressor`;
  - RF comparison uses saved probabilities if available;
  - thresholds are score cut-offs, not calibrated probabilities.
- Check the layout at a narrow browser width and fix overflow.
- Ensure every chart has a title and axis labels.

Tests in `part_1/tests/test_task2.py`:
- Cleaning rules from Prompt 1.
- `confusion_metrics` hand example and real-data threshold check.
- Correlation helper:
  - Pearson matrix is symmetric and has a unit diagonal;
  - Cramér's V values are in `[0, 1]`.
- App smoke test:
  - `create_app(data_path=..., pred_dir=...)` returns an app with a non-empty layout;
  - the layout contains both tab labels;
  - importing the module does not start a server.

Acceptance checks:
```bash
cd part_1
uv run python scripts/clean_dataset.py
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run python run_dashboard.py
```
- Tests pass.
- The app starts with no callback exceptions.
- Manually exercise every control once: table, histogram, correlation (both methods), threshold slider, RF toggle.

Update `part_1/README.md` with:
- what the app does;
- the cleaning command;
- the local run command;
- the test command;
- the environment variables;
- the fact that the app reads saved artefacts and never retrains.

Update `part_1/AI_LOG.md`.
Commit as `test(task2): add dashboard tests and polish`.
````

**Acceptance:** tests pass and the app is usable end to end without console errors.

---

## Prompt 9 — Dockerfile and local container

````markdown
Containerise the dashboard.

Add `gunicorn` as a runtime dependency:
```bash
cd part_1
uv add gunicorn
```

Create `part_1/Dockerfile` with a multi-stage build. Use the repository root as the build context so the raw dataset can be cleaned at build time.

Suggested structure:
1. Builder stage:
   - base image: a pinned `python:3.12-slim-bookworm` tag (or a digest if available);
   - copy a pinned `uv` binary from `ghcr.io/astral-sh/uv` (pin the tag; do not use `latest`);
   - `WORKDIR /app/part_1`;
   - copy `part_1/pyproject.toml`, `part_1/uv.lock`, `part_1/.python-version`, and `part_1/README.md`;
   - `uv sync --frozen --no-dev --no-install-project`;
   - copy `part_1/src`, `part_1/scripts`, `part_1/run_dashboard.py`, and `part_1/data`;
   - copy `datasets/census_income.csv` to `/app/datasets/census_income.csv`;
   - run `uv run python scripts/clean_dataset.py` so `dataset/census_clean.csv` is generated inside the image;
   - `uv sync --frozen --no-dev` to install the project.
2. Runtime stage:
   - same pinned Python slim base;
   - create a non-root user with UID 1000, e.g. `appuser`;
   - copy `/app` from the builder;
   - `WORKDIR /app/part_1`;
   - `ENV PATH="/app/part_1/.venv/bin:$PATH"`;
   - `ENV PORT=8050`, `ENV DASH_DEBUG=false`, `ENV CENSUS_DATA_PATH=/app/part_1/dataset/census_clean.csv`, `ENV PREDICTIONS_DIR=/app/part_1/data`;
   - `EXPOSE 8050`;
   - `USER appuser`;
   - `HEALTHCHECK` that requests `http://127.0.0.1:8050/health`;
   - `CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", "task2.app:server"]`.

Create a repository-root `.dockerignore` that excludes:
- `.git`, `.idea`, `.venv`, `**/.venv`;
- `**/__pycache__`, `**/*.pyc`;
- `part_1/plots`;
- `part_1/dataset/census.csv`;
- `part_1/dataset/census_clean.csv` and `part_1/dataset/census_clean_meta.json` (they are regenerated during the build);
- test caches.

Do not copy the other datasets; only `datasets/census_income.csv` is needed.

Acceptance checks:
```bash
# from the repository root
docker build -f part_1/Dockerfile -t census-dashboard:task2 .
docker run --rm -p 8050:8050 census-dashboard:task2
curl http://127.0.0.1:8050/health
```
- The app is reachable at `http://127.0.0.1:8050`.
- `docker run --rm --entrypoint id census-dashboard:task2` shows UID 1000, not root.
- The image uses the slim base and does not contain compilers or build tools in the runtime stage.
- The app still uses the cleaned CSV generated during the build.

Update `part_1/AI_LOG.md`.
Commit as `build(task2): add multi-stage dashboard image`.
````

**Acceptance:** the image builds, runs as non-root, serves the dashboard, and passes the healthcheck.

---

## Prompt 10 — Docker Compose and optional minikube deployment

````markdown
Add a Docker Compose file and, if time allows, Kubernetes manifests for minikube.

`part_1/docker-compose.yml`:
- service name `dashboard`;
- build context `..` (repository root) and Dockerfile `part_1/Dockerfile`;
- image `census-dashboard:task2`;
- port mapping `8050:8050`;
- environment:
  - `PORT=8050`;
  - `DASH_DEBUG=false`;
  - `CENSUS_DATA_PATH=/app/part_1/dataset/census_clean.csv`;
  - `PREDICTIONS_DIR=/app/part_1/data`;
- `restart: unless-stopped`;
- a healthcheck using the `/health` endpoint;
- do not mount or bake in secrets.

Acceptance:
```bash
cd part_1
docker compose up --build
curl http://127.0.0.1:8050/health
docker compose down
```

Optional Kubernetes manifests in `part_1/k8s/`:
- `configmap.yaml` with the same environment variables;
- `deployment.yaml`:
  - one replica;
  - image `census-dashboard:task2`, `imagePullPolicy: IfNotPresent`;
  - container port 8050;
  - env from the ConfigMap;
  - readiness and liveness probes on `/health`;
  - resource requests and limits;
  - `securityContext`: `runAsNonRoot: true`, `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true` if the app does not need to write at runtime;
- `service.yaml`:
  - selector matching the deployment;
  - port 8050;
  - NodePort `30080` for easy access from minikube.

Minikube checks:
```bash
# from the repository root
minikube image build -t census-dashboard:task2 -f part_1/Dockerfile .
# if your minikube version does not support -f:
# docker build -f part_1/Dockerfile -t census-dashboard:task2 .
# minikube image load census-dashboard:task2

kubectl apply -f part_1/k8s/
kubectl get pods
kubectl rollout status deployment/census-dashboard
minikube service census-dashboard --url
```
- The pod reaches `Running` and `Ready`.
- The service opens the dashboard.
- `kubectl logs` shows no startup errors.
- `kubectl exec` confirms the process is not root.

Document in `part_1/README.md`:
- the Compose command;
- the minikube commands;
- the data-source decision (cleaned at build time, baked into the image);
- the trade-off: simple and reproducible, but changing the data requires a rebuild; a volume or object store would be the alternative.

Update `part_1/AI_LOG.md`.
Commit as `build(task2): add compose and minikube deployment`.
````

**Acceptance:** Compose works locally; the optional minikube deployment is ready and documented.

---

## Prompt 11 — README, Task 2 report, and tag

````markdown
Finish the documentation and tag the task.

Update `part_1/README.md` so a fresh clone can run everything:
1. What Task 2 adds.
2. Architecture and data flow:
   raw CSV -> cleaning script -> `census_clean.csv` -> Dash app;
   Task 1 predictions -> model evaluation tab.
3. Commands:
   - sync dependencies;
   - clean the data;
   - run the tests;
   - run the dashboard;
   - build and run the container;
   - run Compose;
   - optional minikube deployment.
4. Environment variables.
5. The cleaning rules and why categorical features are not one-hot encoded.
6. The fact that the dashboard reads saved artefacts and does not retrain models.
7. Known limitations.

Create `reports/task2.md` from `templates/task_report.md`:
- Replace every placeholder. Delete the guidance blocks.
- Use only facts and numbers you can reproduce from the repository or from commands you actually ran.
- Cover:
  - cleaning rules and their effect on row counts;
  - app architecture and why the saved artefacts are used;
  - why categorical features stay unencoded;
  - how the table, histograms, and correlations work;
  - threshold selection and the RF comparison;
  - image and configuration choices (slim base, pinned dependencies, non-root, env vars, healthcheck);
  - data delivery trade-off (baked at build time vs mounted volume vs download at startup);
  - problems encountered and limitations;
  - next steps.
- Every section must end with an **Evidence** line naming a file, command, test, or screenshot.
- Add at most three screenshots. Save them under `reports/img/` with descriptive names:
  - `task2_explorer_table.png`;
  - `task2_explorer_histogram.png`;
  - `task2_model_threshold.png`.
  Reference them with captions that state a number.
- Do not invent numbers. If a number is missing, run the command and record it.

Update `part_1/AI_LOG.md` with the Task 2 entries.

Final verification from a clean state:
```bash
cd part_1
uv sync --frozen
uv run python scripts/clean_dataset.py
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run python run_dashboard.py
# in another terminal:
curl http://127.0.0.1:8050/health
# from the repository root:
docker build -f part_1/Dockerfile -t census-dashboard:task2 .
docker run --rm -p 8050:8050 census-dashboard:task2
```

Then:
```bash
git add -A
git commit -m "docs(task2): finalise dashboard and deployment"
git tag task2
git push origin main --tags
```

Do not push if the user has not asked you to. Report the final commands and the tag name.
````

**Acceptance:** the README is runnable from a fresh clone, the Task 2 report has no placeholders, all evidence lines are real, and the `task2` tag is created.

---

## Final checklist

- [ ] `part_1/dataset/census_clean.csv` is generated locally and has 48,598 rows.
- [ ] The saved dataset has no one-hot columns and keeps categorical features as strings.
- [ ] `fnlwgt` and `education-num` are absent.
- [ ] `capital-gain == 99999` rows are removed and the removal count is documented.
- [ ] The Data explorer tab shows the table, histograms, and both correlation views.
- [ ] The user can choose the visualisation type and the features to display/compare.
- [ ] The Model evaluation tab changes the confusion matrix as the threshold moves.
- [ ] The RF comparison toggle works at the same threshold, with a documented fallback if probabilities were not added.
- [ ] Unit tests cover cleaning, metrics, correlation, and the app factory.
- [ ] The Docker image builds, runs as non-root, and passes `/health`.
- [ ] Docker Compose works; minikube manifests are present if the advanced path was attempted.
- [ ] `part_1/README.md` explains the app, data flow, commands, and trade-offs.
- [ ] `reports/task2.md` has no placeholders and every section has an Evidence line.
- [ ] `part_1/AI_LOG.md` is up to date.
- [ ] The `task2` tag is created and pushed (only if the user asks).
- [ ] Timebox respected: if time runs out, commit, tag, and stop.
