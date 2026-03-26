import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components.portrayal_components import AgentPortrayalStyle

from models.behavioral_survival.model import NeedsModel


def portrayal(agent):
    if agent is None:
        return None

    style = AgentPortrayalStyle(size=70, marker="o", zorder=2)

    if agent.is_dead:
        style.update(("color", "lightgray"))
        style.update(("size", 40))
        return style

    colors = {
        "hide": "black",
        "rest": "tab:blue",
        "search_food": "tab:red",
        "wander": "gray",
    }
    style.update(("color", colors.get(agent.last_action, "gray")))
    return style


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


model_params = {
    "n_agents": Slider("Agents", 60, 20, 120, 5),
    "n_food": Slider("Food Cells", 30, 10, 80, 2),
    "n_danger": Slider("Danger Cells", 18, 5, 50, 1),
    "hunger_weight": Slider("Hunger Weight", 1.0, 0.2, 2.0, 0.1),
    "fear_weight": Slider("Fear Weight", 1.1, 0.2, 2.0, 0.1),
    "low_energy_weight": Slider("Energy Weight", 0.9, 0.2, 2.0, 0.1),
    "wander_bias": Slider("Wander Bias", 0.08, 0.0, 0.5, 0.01),
    "temperature": Slider("Policy Temperature", 0.12, 0.02, 1.0, 0.01),
    "score_scale": Slider("Score Scale", 6.0, 1.0, 15.0, 0.5),
    "panic_threshold": Slider("Panic Threshold", 0.45, 0.1, 0.9, 0.05),
    "contagion_strength": Slider("Contagion Strength", 0.06, 0.0, 0.3, 0.01),
    "hide_interrupt_threshold": Slider("Hide Interrupt Threshold", 0.85, 0.4, 1.0, 0.05),
}

model = NeedsModel()

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(portrayal)
renderer.post_process = post_process_space
renderer.draw_agents()

behavior_plot = make_plot_component(
    {
        "Hiding": "black",
        "Resting": "tab:blue",
        "Searching Food": "tab:red",
        "Wandering": "gray",
    }
)

needs_plot = make_plot_component(
    {
        "Avg Hunger": "tab:orange",
        "Avg Fear": "tab:purple",
        "Avg Energy": "tab:green",
    }
)

score_plot = make_plot_component(
    {
        "Avg Hide Score": "black",
        "Avg Rest Score": "tab:blue",
        "Avg Search Score": "tab:red",
        "Avg Wander Score": "gray",
    }
)

prob_plot = make_plot_component(
    {
        "Avg Hide Prob": "black",
        "Avg Rest Prob": "tab:blue",
        "Avg Search Prob": "tab:red",
        "Avg Wander Prob": "gray",
    }
)

weight_plot = make_plot_component(
    {
        "Avg Hunger Weight": "tab:orange",
        "Avg Fear Weight": "tab:purple",
        "Avg Energy Weight": "tab:green",
    }
)

survival_plot = make_plot_component(
    {
        "Survival Rate": "tab:green",
    }
)

environment_plot = make_plot_component(
    {
        "Total Food Consumed": "tab:orange",
        "Total Danger Exposure": "tab:red",
    }
)

regime_plot = make_plot_component(
    {
        "Survival Regime": "tab:red",
        "Fear Regime": "black",
        "Recovery Regime": "tab:blue",
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[
        behavior_plot,
        needs_plot,
        score_plot,
        prob_plot,
        weight_plot,
        survival_plot,
        environment_plot,
        regime_plot,
    ],
    model_params=model_params,
    name="Behavior Engine Prototype: Needs-Based Survival Domain",
)

page