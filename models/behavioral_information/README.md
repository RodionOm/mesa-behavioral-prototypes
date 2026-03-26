# Information Behavior Model

## Description

Agents interact with information through behavioral strategies:

- sharing
- verifying
- ignoring
- exploring

Each agent has:

- curiosity
- confidence
- social pressure

---

## Behavioral Mechanics

Same pipeline as survival model:
scores → softmax → action → update → regime


---

## Key Features

- non-spatial interaction (social influence)
- dynamic confidence evolution
- regime switching:
  - broadcast
  - cautious
  - passive

---

## Observed Behavior

- dominant strategies emerge (e.g. sharing bias)
- curiosity decays over time
- social pressure drives convergence

---

## Limitations

- strong dominance of certain actions
- requires manual tuning
- regime transitions are heuristic

---

## Why This Model Matters

This model shows:

- framework is reusable across domains
- but behavior outcomes are domain-sensitive
- abstraction does not remove need for tuning

---

## Conclusion

> The same behavior engine produces fundamentally different dynamics depending on domain structure.