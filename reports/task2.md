# Task 2 report: App deployment

| Field | Value |
| --- | --- |
| Student | Stefano Negri Merlo |
| Dataset | Adult Census Income (`census_income.csv`) |
| Task | 2|
| Tag / commit | `task2`|
| How to run | `cd part_1 && docker build --tag 'task2' . && docker run -p 8050:8050 task2:latest` |
| Timebox | 2h (finished) |

## 1. Summary

I have created a dockerized app for visualizing data and the results of the models.

**Evidence:** `part_1/src/task2`, `part_1/run_dashboard.py`.

## 2. Goal and success criteria

**Goal.** Task 2 requires the creation of a dockerized interactive visualization app for the dataset. Additionally I have provided visualization for the models of Task 1 as well.

| Criterion | How it is measured | Target | Result |
| --- | --- | --- | --- |
| Presence of visualization widgets | Manual verification | Table, histograms, correlation, threshold and metrics | Passed |
| Dockerized | Manual verification | Docker container created | Passed |

**Evidence:** `part_1/run_dashboard.py` and `part_1/Dockerfile`

## 3. Approach and key decisions

| # | Finding | Decision | Alternative considered | Where in the code |
| --- | --- | --- | --- | --- |
| D1 | Need a visualization framework | Use python dash | React app | `part_1/src/task2/app.py`|
| D2 | In task 1 I have directly trained a model instead of analyzing data | Keep task 1 as it is and display both data analysis and model results| Start from the beginning| `part_1/run_dashboard.py`|

**Evidence:** `part_1/run_dashboard.py`, `part_1/src/task2`

## 4. Results

N/A

## 5. Problems and how you handled them

| Problem | Cause | What you did | Status |
| --- | --- | --- | --- |
| Not enough time to complete the task by hand | Too much time spent on task 1 | Used a coding agent| Workaround |

**Evidence:** `reports/task2_session.json`, `reports/AI_LOG.md`, `.opencode/plan/task2_plan.md`

## 6. Limitations and trade-offs

N/A

## 7. Next steps

| Priority | Next step | Why | Effort |
| --- | --- | --- | --- |
| 1 | Polish the app | It is clearly made by an AI agent and it needs human attention to details | M |

## 8. Use of AI assistants

| Tool | What you used it for | Where (files or sections) | How you checked the output |
| --- | --- | --- | --- |
| AI coding assistant | The entire task| `../part_1/src/task2` | By manually testing the app |

## Checklist

- [x] Task tag pushed and named in the table at the top
- [x] Runs from a fresh clone with the commands above
- [x] No secrets, API keys or large data files committed
- [x] Every section has an Evidence line
- [x] Every figure has a caption with a number
- [x] Guidance blocks and placeholders removed
- [x] `part_1/AI_LOG.md` up to date
