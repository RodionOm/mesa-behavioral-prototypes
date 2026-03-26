# mesa-behavioral-prototypes

Behavioral modeling prototypes built in Mesa as preparation for GSoC 2026.

---

## Overview

This repository explores how reusable behavioral abstractions (actions, policies, regimes) can be applied across different simulation domains in Mesa.

The goal is not to build polished simulations, but to:

- identify reusable behavioral patterns
- test abstraction boundaries
- expose friction points in Mesa’s current API
- evaluate how far we can push general-purpose behavior modeling

---

## Core Idea

Most Mesa models reimplement the same logic:

- action selection
- decision heuristics
- behavioral state transitions
- regime switching
- history / memory effects

This project extracts those into a **reusable framework layer**:

- `Action` → atomic behavior unit
- `SoftmaxPolicy` → stochastic decision mechanism
- `BehaviorMixin` → agent-level decision pipeline

Then validates them across multiple domains.

---

## Repository Structure


framework/
action.py # Base Action abstraction
policy.py # Softmax-based decision policy
behavior.py # Reusable agent behavior pipeline

models/
behavioral_survival/
model.py
actions.py
app.py
README.md

behavioral_information/
    model.py
    actions.py
    app.py
    README.md

notes/
comparison_notes.md
framework_notes.md


---

## Models

### 🧠 Needs-Based Survival
A multi-drive agent system with:

- hunger / fear / energy competition
- regime switching (survival / recovery / panic)
- spatial environment (food + danger)
- emergent survival dynamics

➡️ Strong validation of behavior abstraction layer.

---

### 🌐 Information Behavior Model
Agents interacting with information through:

- sharing / verifying / ignoring / exploring
- confidence / curiosity / social pressure
- broadcast / cautious / passive regimes

➡️ Cross-domain validation with different dynamics.

---

## Key Insight

The same behavioral pipeline works across domains:


scores → softmax → action → state update → regime shift


BUT:

- domain-specific scoring is still manual
- regime logic is duplicated
- diagnostics are ad-hoc

This indicates **partial abstraction success** but also clear extension points.

---

## Running the Models

From project root:

```bash
source .venv/bin/activate

solara run models/behavioral_survival/app.py
solara run models/behavioral_information/app.py