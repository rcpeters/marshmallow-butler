# Marshmallow Butler ☕🤖

A fun robotics project built to turn a desktop robot arm into a **Hot Cocoa Marshmallow Butler** for kids—ask for a marshmallow, and the robot spots one on the table, picks it up gently with compliant TPU fingers, clears the mug rim, drops it right into the hot cocoa, and checks its own work! Zero custom model training required.

*(Cocoa-safety feature: it drops cleanly above the rim and folds neatly back out of the way—no dipped robot fingers in the chocolate!)*

---

## What's Inside (The Tech Stack)

- **Robot Arm:** **LeRobot SO-101** Follower arm (6-DOF Feetech STS3215 servos + soft TPU grippers for gentle marshmallow grabs)
- **Edge Computer:** **Arduino VENTUNO Q** running a lean ARM64 Docker stack with USB webcam vision
- **Local Voice Gatekeeper:** **Local Gemma LLM** (`gemma-4-e2b`) running on the edge to filter casual kitchen chatter and confirm real drink requests before waking the cloud
- **Cloud Spatial Brain:** **Gemini 4 Hydrogen** (`gemini-4-hydrogen`) for on-demand single-frame vision, bounding-box object width estimation, cup targeting, and post-drop verification

---

## How Marshmallow Delivery Works

1. **"Can I get a marshmallow in my cocoa?"** — The local **Gemma** gatekeeper decides whether it's an actual cocoa request (ignoring kids yelling about everything else in the room).
2. **Spot & Measure (Cloud):** One overhead webcam snapshot goes on-demand to **Gemini 4 Hydrogen**, which checks safety (`cocoa` mode for food treats only, or `--toy-mode` for general small tabletop objects with narrow-axis wrist roll + height estimation) and targets the cup.
3. **Pick & Plop (Edge):** Using 4-corner ArUco homography + camera height parallax correction, the **Arduino VENTUNO Q** plans a cup-safe trajectory for the **SO-101**—closing TPU jaws proportionally to item width, lifting high over the mug rim, releasing cleanly, and folding neatly home.
4. **Did It Land?** **Gemini 4 Hydrogen** takes one quick look after the arm folds away to verify the drop landed in the cup.

---

## Requirements & Setup

- **Python dependencies:** See [requirements.txt](requirements.txt)
- **Architecture & hardware guide:** See [PLAN.md](PLAN.md)
- **Calibration & servo diagnostics:** See [UTILITIES.md](UTILITIES.md)
- **Hardware & workspace tuning:** See [config/hardware.yaml](config/hardware.yaml)

### Quick Run (Docker on VENTUNO Q)

```bash
cp .env.example .env   # Add your GEMINI_API_KEY and verify serial/camera device paths
docker compose build robot-controller
docker compose run --rm robot-tools pytest -v

# Calibrate arm & ArUco mat, then launch the CLI butler:
docker compose run --rm robot-tools python3 scripts/calibrate_arm.py
docker compose run --rm robot-tools python3 scripts/calibrate_mat.py
docker compose run --rm robot-tools python3 -m src.main --mode cli

# Optional: launch directly in Toy Mode for general small objects (or type 'toy mode' / 'cocoa mode' at the prompt):
docker compose run --rm robot-tools python3 -m src.main --mode cli --toy-mode
```
