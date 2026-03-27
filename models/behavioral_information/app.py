import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components.portrayal_components import AgentPortrayalStyle

from models.behavioral_information.model import InformationBehaviorModel


def portrayal(agent):
    if agent is None:
        return None

    style = AgentPortrayalStyle(size=70, marker="o", zorder=2)

    colors = {
        "share": "tab:red",
        "verify": "tab:blue",
        "ignore": "gray",
        "explore": "tab:green",
    }

    style.update(("color", colors.get(agent.last_action, "gray")))
    return style


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


model_params = {
    "n_agents": Slider("Agents", 50, 20, 100, 5),
    "curiosity_weight": Slider("Curiosity Weight", 1.0, 0.2, 2.0, 0.1),
    "confidence_weight": Slider("Confidence Weight", 1.0, 0.2, 2.0, 0.1),
    "social_weight": Slider("Social Weight", 1.0, 0.2, 2.0, 0.1),
    "curiosity_drift": Slider("Curiosity Drift", 0.01, -0.05, 0.05, 0.005),
    "confidence_drift": Slider("Confidence Drift", -0.003, -0.05, 0.05, 0.005),
    "temperature": Slider("Policy Temperature", 0.14, 0.02, 1.0, 0.01),
    "score_scale": Slider("Score Scale", 5.0, 1.0, 15.0, 0.5),
    "memory_penalty_weight": Slider("Memory Penalty", 0.15, 0.0, 0.5, 0.01),
    "verify_interrupt_threshold": Slider("Verify Interrupt Threshold", 0.22, 0.05, 0.50, 0.01),
}

model = InformationBehaviorModel()

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(portrayal)
renderer.post_process = post_process_space
renderer.draw_agents()

behavior_plot = make_plot_component(
    {
        "Sharing": "tab:red",
        "Verifying": "tab:blue",
        "Ignoring": "gray",
        "Exploring": "tab:green",
    }
)

state_plot = make_plot_component(
    {
        "Avg Curiosity": "tab:orange",
        "Avg Confidence": "tab:purple",
        "Avg Social Pressure": "tab:brown",
        "Avg Topic Salience": "tab:olive",
    }
)

score_plot = make_plot_component(
    {
        "Avg Share Score": "tab:red",
        "Avg Verify Score": "tab:blue",
        "Avg Ignore Score": "gray",
        "Avg Explore Score": "tab:green",
    }
)

prob_plot = make_plot_component(
    {
        "Avg Share Prob": "tab:red",
        "Avg Verify Prob": "tab:blue",
        "Avg Ignore Prob": "gray",
        "Avg Explore Prob": "tab:green",
    }
)

regime_plot = make_plot_component(
    {
        "Broadcast Regime": "tab:red",
        "Cautious Regime": "tab:blue",
        "Passive Regime": "gray",
    }
)

volume_plot = make_plot_component(
    {
        "Total Shares": "tab:red",
        "Total Verifications": "tab:blue",
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[
        behavior_plot,
        state_plot,
        score_plot,
        prob_plot,
        regime_plot,
        volume_plot,
    ],
    model_params=model_params,
    name="Behavior Engine Prototype: Information Domain",
)

page