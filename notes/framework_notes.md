# Framework Layer Notes

## Components

### Action
Encapsulates:

- scoring logic
- execution logic

---

### SoftmaxPolicy

Handles:

- score normalization
- stochastic selection
- exploration via temperature

---

### BehaviorMixin

Provides:

- decision pipeline
- history tracking
- persistence & fatigue
- regime handling hooks

---

## Design Principles

- separation of concerns
- composability
- domain-agnostic core

---

## Limitations

- scoring functions still manual
- regime logic duplicated
- no standard diagnostics interface

---

## Key Insight

> The hardest part is not selecting actions, but defining meaningful scores.

---

## Future Work

- plug-in scoring modules
- declarative behavior configs
- built-in diagnostics

---

## Conclusion

The framework is viable, but incomplete → ideal GSoC scope.