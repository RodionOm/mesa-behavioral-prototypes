# mesa-behavioral-prototypes

Behavioral modeling prototypes built in Mesa as preparation for GSoC 2026.

---

## Overview

This repository explores how reusable behavioral abstractions (actions, policies, regimes) can be applied across different simulation domains in Mesa.

The goal is not to build polished simulations, but to:

- identify reusable behavioral patterns
- test abstraction boundaries
- expose friction points in Mesa’s current API
- evaluate how far general-purpose behavior modeling can go

---

## Core Idea

Most Mesa models repeatedly implement the same logic:

- action selection
- decision heuristics
- behavioral state transitions
- regime switching
- memory / history effects

This project extracts these into a **reusable behavior layer**:

- `Action` → atomic unit of behavior (score + execution)
- `SoftmaxPolicy` → stochastic decision mechanism
- `BehaviorMixin` → reusable decision pipeline for agents

These abstractions are then validated across different domains.

---

## Why This Is Not Just Another Mesa Model

This repository focuses on **behavior abstraction**, not domain modeling.

Instead of building a single simulation, it explores:

- what parts of agent logic are reusable
- where Mesa forces manual duplication
- how decision pipelines can be standardized

This makes it closer to a **framework exploration** than a model implementation.

---

## Repository Structure
framework/
action.py # Base Action abstraction
policy.py # Softmax decision policy
behavior.py # Agent behavior pipeline

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

requirements.txt


---

## Models

### 🧠 Needs-Based Survival

A multi-drive agent system where agents balance:

- hunger
- fear
- energy

Environment includes:

- food sources
- danger zones

Features:

- competing drives
- regime switching
- spatial interaction
- stochastic policy
- emergent survival dynamics

➡️ Strong validation of behavior abstraction layer.

---

### 🌐 Information Behavior Model

Agents interact with information through:

- sharing
- verifying
- ignoring
- exploring

Internal states:

- curiosity
- confidence
- social pressure

Features:

- social influence dynamics
- regime switching (broadcast / cautious / passive)
- stochastic behavior selection

➡️ Cross-domain validation of the same behavioral pipeline.

---

## Key Insight

The same behavioral pipeline works across domains:


scores → softmax → action → state update → regime shift


However:

- scoring functions remain domain-specific
- regime logic is duplicated
- diagnostics are ad-hoc

This indicates **partial abstraction success**, with clear directions for generalization.

---

## Running the Models

```bash
source .venv/bin/activate

solara run models/behavioral_survival/app.py
solara run models/behavioral_information/app.py

```

Author:

Rodion Omelich
GSoC 2026 Applicant