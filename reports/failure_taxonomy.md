# Failure Taxonomy

This file is generated from local sample baseline trajectories.

| Code | Count | Meaning | Example |
|---|---:|---|---|
| F1_TOOL_NAME_ERROR | 2 | Tool name is missing, wrong, or unsupported. | carbench:car_005_unhandled_massage - Expected tool sequence ['seat.set_massage'], got [].<br>bfcl:bfcl_004_temperature_convert - Expected tool sequence ['temperature.convert'], got []. |
| F2_ARGUMENT_ERROR | 0 | Tool arguments are missing, malformed, or wrong. | - |
| F3_STATE_TRACKING_ERROR | 0 | Multi-turn state or slot values are inconsistent. | - |
| F4_MISSING_CLARIFICATION | 0 | Ambiguous request should have been clarified. | - |
| F5_CAPABILITY_HALLUCINATION | 0 | Model invents an unavailable tool, capability, or result. | - |
| F6_SAFETY_BOUNDARY_ERROR | 0 | Model crosses a safety or capability boundary. | - |
| F7_PLANNING_ORDER_ERROR | 0 | Tool sequence or task plan order is wrong. | - |
| F8_VERBOSE_OR_LOOP | 0 | Response is unnecessarily long, repetitive, or loops. | - |
