# 📄 2. behavioral_survival/README.md

```md
# Needs-Based Survival Model

## Description

A behavioral simulation where agents balance:

- hunger
- fear
- energy

Agents operate in a spatial environment with:

- food sources
- danger zones

---

## Behavioral Mechanics

Agents choose actions via:

- scoring functions
- softmax selection
- persistence + fatigue penalties

Actions:

- `hide`
- `rest`
- `search_food`
- `wander`

---

## Key Features

- competing internal drives
- regime switching
- spatial interaction
- stochastic policy
- survival dynamics

---

## Why This Model Matters

This model validates that:

- behavior abstraction works in complex environments
- softmax policy produces realistic trade-offs
- regime logic emerges from state thresholds

---

## Observed Behavior

- agents oscillate between survival and recovery
- fear spreads locally
- resource scarcity creates pressure cycles

---

## Conclusion

This model demonstrates that:

> A reusable behavior layer can support complex adaptive systems in Mesa.