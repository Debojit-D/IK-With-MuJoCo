# IK-With-MuJoCo

This repository accompanies the hands-on robotics workshop  
**“Hands-On Inverse Kinematics with MuJoCo.”**

The workshop introduces the fundamentals of **forward and inverse kinematics** and provides step-by-step, practical implementations of inverse kinematics on the **Franka Emika manipulator** using three complementary approaches:

- **Damped Least Squares (DLS) Inverse Kinematics**, implemented from scratch  
- **Library-based Inverse Kinematics** using MuJoCo’s `mink` framework  
- **Optimization-based Inverse Kinematics**, formulated and solved as a **Quadratic Program (QP)**  

The goal of this repository is to bridge theory and practice by guiding participants through intuitive explanations and hands-on implementations in simulation.

---

## Workshop Website

All workshop material, theory notes, and guided instructions are available here:  
**https://debojit.notion.site/Hands-On-Inverse-Kinematics-with-MuJoCo-2e8f4aa874c48012adc4e546edf7877e**

---

## Setup

Minimal installation is required. The workshop is best experienced on **Ubuntu**, but the code is compatible with **Windows and macOS** as well.

```bash
pip install mujoco loop_rate_limiters mink
