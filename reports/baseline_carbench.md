# CARBENCH Baseline Report

## Run

- Adapter: `local_rules`
- Samples: 5
- Samples with failures: 1

## Metrics

| Metric | Value |
|---|---:|
| success | 0.8000 |
| tool_accuracy | 0.8000 |
| tool_name_accuracy | 0.8000 |
| argument_accuracy | 0.8000 |
| executable_tool_rate | 1.0000 |
| hallucination_rate | 0.0000 |
| state_consistency | 1.0000 |
| disambiguation_success | 1.0000 |
| avg_turns | 1.0000 |

## Failure Counts

| Code | Count | Meaning |
|---|---:|---|
| F1_TOOL_NAME_ERROR | 1 | Tool name is missing, wrong, or unsupported. |

## Notes

These are local sample smoke-test results, not official benchmark scores.
