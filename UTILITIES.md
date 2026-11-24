# SO-101 Utilities & Servo Diagnostics Guide

This guide covers all setup, inspection, and troubleshooting utilities in `scripts/`—especially how to check every joint angle safely (with torque **OFF**) and tell whether a servo needs a **software sign flip** vs. a **physical horn/servo re-mount**.

> **Safety Rule:** Every command in [scripts/check_joints.py](file:///c:/Users/info/Documents/git/so101-ventuno-q/scripts/check_joints.py) disables servo torque before reading, so you can move the arm freely by hand with zero motor motion.

---

## 1. Quick Command Reference

Run all utilities from the repo root on your VENTUNO Q using the `robot-tools` container:

| Utility | Command | Purpose |
| :--- | :--- | :--- |
| **Port Finder** | `docker compose run --rm robot-tools python3 scripts/find_port.py` | Find which `/dev/ttyACM*` belongs to your SO-101 adapter |
| **Motor ID Setup** | `docker compose run --rm robot-tools python3 scripts/setup_motors.py --port /dev/ttyACM0` | Program STS3215 IDs `1–6` and 1 Mbps baudrate one servo at a time |
| **Arm Calibration** | `docker compose run --rm robot-tools python3 scripts/calibrate_arm.py` | Record upright `zero_tick` + safe `[min_tick, max_tick]` bounds |
| **Joint Audit** | `docker compose run --rm robot-tools python3 scripts/check_joints.py` | Ping all 6 servos, print live angles, and check for 0/4095 wrap risk |
| **Live Angle Monitor** | `docker compose run --rm robot-tools python3 scripts/check_joints.py --live` | Stream live joint angles (`deg`/`%`) and raw ticks while moving by hand |
| **Capture Folded Home** | `docker compose run --rm robot-tools python3 scripts/check_joints.py --capture-home` | Record your arm's resting folded pose into `arm.home_pose_deg` |
| **Servo Flip Wizard** | `docker compose run --rm robot-tools python3 scripts/check_joints.py --check-signs` | Detect `+1.0` vs `-1.0` polarity for each joint and print `joint_signs` YAML |
| **Mat Calibration (Camera)** | `docker compose run --rm robot-tools python3 scripts/calibrate_mat.py` | Detect ArUco markers `1–4` and save pixel-to-base homography |
| **Mat Verify & Calibrate (Arm)** | `docker compose run --rm robot-tools python3 scripts/calibrate_mat.py --arm-calibrate` | Move/jog arm over reachable markers to align & calibrate the mat |
| **Verify Single Marker** | `docker compose run --rm robot-tools python3 scripts/calibrate_mat.py --verify-marker 1` | Move closed gripper tip over marker `1` to verify alignment |
| **Gemini Pre-Grasp Test** | `docker compose run --rm robot-tools python3 scripts/test_gemini_point.py --move-pregrasp` | Point with `gemini-4-hydrogen` and hover 5 cm above target item |

---

## 2. Checking Joint Angles & Detecting Backwards Servos

### Step A — Run the Audit & Wrap-Boundary Check
```bash
docker compose run --rm robot-tools python3 scripts/check_joints.py
```
Look at two things in the output:
1. **`[OK]` vs `[WARN_NEAR_WRAP]`:**
   - Feetech STS3215 magnetic encoders read `0` to `4095` (`360°`), with **`2048`** as the exact midpoint.
   - If `[min_tick, max_tick]` gets close to `0` or `4095` (`< 400` or `> 3695`), the encoder is dangerously close to wrapping around (`4095 -> 0`).
2. **Current angles (`now=... deg`):**
   - Hold the arm straight upright like a flagpole (`0°` pose). Every arm joint (`shoulder_pan` through `wrist_roll`) should read close to **`0°`**.
   - Use `--live` to watch angles update continuously as you move each joint:
     ```bash
     docker compose run --rm robot-tools python3 scripts/check_joints.py --live
     ```

---

### Step B — Do I Need to Physically Flip a Servo, or Just Change `joint_signs`?

There are **two different kinds** of "installed backwards" on an SO-101—only one requires turning a screwdriver:

#### Case 1: Servo Horn / Body Is Offset ~180° (Near `0` or `4095` Wrap) — **PHYSICALLY RE-SEAT HORN**
- **Symptoms:**
  - `check_joints.py` prints `[WARN_NEAR_WRAP]` (`min_tick < 400` or `max_tick > 3695`), OR
  - In the upright `0°` pose, a joint's `zero_tick` is far from center (e.g., `< 600` or `> 3500`), OR
  - Sweeping the joint jumps suddenly between `~0` and `~4095`.
- **Why software can't fix this:** The encoder's physical wrap seam (`0 <-> 4095`) sits inside your working range, so `min_tick`/`max_tick` breaks.
- **Fix:**
  1. Power off / leave torque OFF.
  2. Unscrew the servo horn (or remove the joint bracket), run `check_joints.py --live`, rotate the **unattached** servo shaft until its raw tick reads near **`2048`**, then re-bolt the arm link in its **straight upright** position.
  3. Re-run `python3 scripts/calibrate_arm.py`.

#### Case 2: Joint Moves the Opposite Direction (`+` vs `-`), Range Is Safely Centered — **SOFTWARE FIX ONLY (No Disassembly)**
- **Symptoms:**
  - `check_joints.py` prints `[OK]` (all ticks safely inside `~400..3695`), **but** commanding a positive IK angle moves the joint the opposite way (e.g., tilting `shoulder_lift` forward produces negative ticks instead of positive ticks).
- **Why this happens:** Depending on which side of the arm a servo faces (e.g. `shoulder_lift` facing left vs `elbow_flex` facing right), raw encoder ticks naturally go negative in one direction and positive in the other. That's completely normal on SO-101!
- **Fix (1 minute with the Wizard):**
  Run:
  ```bash
  docker compose run --rm robot-tools python3 scripts/check_joints.py --check-signs
  ```
  Follow the prompts one joint at a time. At the end it prints the exact `joint_signs:` block to paste into [config/hardware.yaml](file:///c:/Users/info/Documents/git/so101-ventuno-q/config/hardware.yaml) under `kinematics:`.

---

## 3. Expected Physical Behavior per Joint (Quick Verification Table)

When running `--live` from the **straight upright** pose (`0°` everywhere, gripper half-open):

| Motor ID | Joint | Physical Motion Checked | Expected IK Positive Sense | Notes |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `shoulder_pan` | Rotate base **LEFT** (`+y` when facing mat) | `+` in IK (`joint_signs` handles `-` encoder) | Center `zero_tick` ~2048 facing dead ahead over mat |
| **2** | `shoulder_lift` | Tilt upper arm **FORWARD** toward mat | `+` in IK (`joint_signs` handles `-` encoder) | Check cable slack when tilting all the way back/forward |
| **3** | `elbow_flex` | Bend forearm **FORWARD / DOWN** | `+` in IK | Must reach `~+140°` without hitting `0`/`4095` wrap |
| **4** | `wrist_flex` | Pitch gripper **DOWN** toward mat | `+` in IK | Straight along forearm = `0°`; 90° down = `+90°` |
| **5** | `wrist_roll` | Rotate gripper **CW** looking toward fingers | `+` in IK | Keep `zero_tick` away from `0/4095` if twisting ±90° |
| **6** | `gripper` | **OPEN** jaws wide | `max_tick > min_tick` (`100%` open, `0%` closed) | If `open < closed`, re-seat horn or check servo orientation |

---

## 4. What to Check If `check_joints.py` Shows Everything `[OK]`

Once all 6 joints pass `check_joints.py` (`[OK]` margins + verified `joint_signs`):
1. Run `python3 scripts/calibrate_arm.py` starting from the **straight upright flagpole pose**.
2. Verify `kinematics` link lengths (`base_height_m`, `upper_arm_m`, `forearm_m`, `wrist_to_tip_m`) in [config/hardware.yaml](file:///c:/Users/info/Documents/git/so101-ventuno-q/config/hardware.yaml) match ruler measurements on your arm.
3. Run `python3 scripts/test_gemini_point.py --move-pregrasp` to confirm a clean **5 cm** hover above the target before descending to grasp.
