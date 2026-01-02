# Adaptive Traffic Signal Control Using Reinforcement Learning

## Overview

This project implements an **adaptive traffic signal control system** using **Deep Reinforcement Learning (DQN)** and compares it against a **fixed-time traffic signal controller**.[conversation_history:1] The environment simulates a four-way intersection with cars and buses, and the agent learns to minimize congestion by adjusting signal phases dynamically based on real-time traffic conditions.[conversation_history:1] The system includes a custom traffic simulation environment, a PyTorch-based DQN agent, a fixed-time baseline controller, and real-time visualization with training metrics such as reward, queue length, and waiting time.[conversation_history:1]

---

## Features

- Simulation of a single four-way intersection with:
  - Multiple vehicle types (cars, buses).[conversation_history:1]
  - Dynamic vehicle spawning and movement.[conversation_history:1]
- **Reinforcement Learning controller**:
  - DQN with experience replay and a target network.[conversation_history:1]
  - Epsilon-greedy exploration with adjustable epsilon.[conversation_history:1]
- **Fixed-time controller** baseline:
  - Periodic phase switching with configurable green duration.[conversation_history:1]
- Reward function balancing:
  - Total waiting time.[conversation_history:1]
  - Queue length.[conversation_history:1]
  - Throughput (vehicles passed).[conversation_history:1]
  - Queue balance across directions.[conversation_history:1]

---
