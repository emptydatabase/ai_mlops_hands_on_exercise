# Task 2 report: App deployment

| Field | Value |
| --- | --- |
| Student | Stefano Negri Merlo |
| Dataset | Adult Census Income (`census_income.csv`) |
| Task | 3|
| Tag / commit | `task3`|
| How to run | `cd part_2 && uv run python run.py` |
| Timebox | 1h (not finished) |

## 1. Summary

I have created a simple agent that answers questions about the database

**Evidence:** `part_2/src/task3`, `part_2/run.py`.

## 2. Goal and success criteria

**Goal.** Task 2 requires the creation of a dockerized interactive visualization app for the dataset. Additionally I have provided visualization for the models of Task 1 as well.

| Criterion | How it is measured | Target | Result |
| --- | --- | --- | --- |
| Answer simple questions about the dataset | Manual verification | Can answer at least one question correcly | Passed |
| Use the tools you give it to answer questions about the dataset and Part 1 analysis | Test suite | Uses at least one tool correctly | Not done |
| Use advanced tools, such as a prediction or classification model, to make predictions | Test suite | Uses at least one tool correctly | Not done |
| Ship the agent as a CLI tool. | Manual verification | It runs correctly | Passed |
| Ship the agent as a containerised service with Docker or Kubernetes| Manual verification | It runs correctly | Not done |

**Evidence:** `part_2/run.py` and `part_2/src/task3`

## 3. Approach and key decisions

N/A

## 4. Results

N/A

## 5. Problems and how you handled them

| Problem | Cause | What you did | Status |
| --- | --- | --- | --- |
| No API key providec | No API key providec | Used local ollama service | Workaround |

**Evidence:** `part_2/src/task3/agent.py`

## 6. Limitations and trade-offs

N/A

## 7. Next steps

| Priority | Next step | Why | Effort |
| --- | --- | --- | --- |
| 1 | Add tools | It is needed for basic functionalities of the agent | L |
| 2 | Connect it to the trained model | It is needed for advanced functionalities of the agent | L |

## 8. Use of AI assistants

N/A

## Checklist

- [x] Task tag pushed and named in the table at the top
- [x] Runs from a fresh clone with the commands above
- [x] No secrets, API keys or large data files committed
- [x] Every section has an Evidence line
- [x] Every figure has a caption with a number
- [x] Guidance blocks and placeholders removed
- [x] `part_2/AI_LOG.md` up to date
