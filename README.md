# Adaptive Traffic Signal Control Using Reinforcement Learning

## Overview

This project implements an **adaptive traffic signal control system** using **Deep Reinforcement Learning (DQN)** and compares it against a **fixed-time traffic signal controller**. The environment simulates a four-way intersection with cars and buses, and the agent learns to minimize congestion by adjusting signal phases dynamically based on real-time traffic conditions. The system includes a custom traffic simulation environment, a PyTorch-based DQN agent, a fixed-time baseline controller, and real-time visualization with training metrics such as reward, queue length, and waiting time.

---

## Features

- Simulation of a single four-way intersection with:
  - Multiple vehicle types (cars, buses).
  - Dynamic vehicle spawning and movement.
- **Reinforcement Learning controller**:
  - DQN with experience replay and a target network.
  - Epsilon-greedy exploration with adjustable epsilon.
- **Fixed-time controller** baseline:
  - Periodic phase switching with configurable green duration.
- Reward function balancing:
  - Total waiting time.
  - Queue length.
  - Throughput (vehicles passed).
  - Queue balance across directions.
- Reduced congestion ~10% vs fixed-time baseline and improved intersection througput.
- Built the simulation environment and real-time visualization using PyGame, enabling iterative training and policy evaluation.
