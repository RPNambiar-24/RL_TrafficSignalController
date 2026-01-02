
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pygame
import random
import sys
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Set device for GTX 1660 Ti optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Traffic Signal Control Environment
class TrafficEnvironment:
    def __init__(self):
        

        # Environment parameters optimized for realistic traffic
        self.max_cars_per_lane = 20
        self.spawn_probability = {'car': 0.3, 'bus': 0.1}  # Different spawn rates
        self.max_green_time = 60
        self.min_green_time = 10
        self.yellow_time = 3
        self.all_red_time = 2

        # State space: [phase, time_in_phase, queue_lengths(8), waiting_times(8), approaching_vehicles(8)]
        self.state_size = 25
        self.action_size = 4  # North-South Green, East-West Green, All Red, Emergency

        # Directions: North, East, South, West
        self.directions = ['north', 'south', 'east', 'west']
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.vehicles = {
            'north': {'cars': [], 'buses': []},
            'south': {'cars': [], 'buses': []},
            'east': {'cars': [], 'buses': []},
            'west': {'cars': [], 'buses': []}
        }

        self.current_phase = 0  # 0: NS Green, 1: EW Green
        self.phase_timer = 0
        self.total_vehicles_passed = 0
        self.total_waiting_time = 0
        self.step_count = 0

        return self._get_state()

    def _get_state(self):
        """Get current state representation as a numpy array of length 25"""
        state = []

        # Normalize current phase over [0,3] because possible phases: 0,1,2,3 (assuming 4 phases)
        state.append(float(self.current_phase) / 3.0)

        # Normalize time in current phase by max_green_time
        state.append(float(self.phase_timer) / self.max_green_time)

        # Queue lengths (4 directions) - each normalized by max_cars_per_lane
        for direction in self.directions:
            cars = len(self.vehicles[direction]['cars'])
            buses = len(self.vehicles[direction]['buses'])
            # Each bus counts as 2 cars for weighting
            total_queue = (cars + 2 * buses) / float(self.max_cars_per_lane)
            state.append(min(total_queue, 1.0))
    
        # Average waiting times (4 directions) normalized, avoid division by zero
        for direction in self.directions:
            total_vehicles = len(self.vehicles[direction]['cars']) + len(self.vehicles[direction]['buses'])
            if total_vehicles > 0:
                total_waiting_sum = sum(v['waiting_time'] for v in self.vehicles[direction]['cars'])
                total_waiting_sum += sum(v['waiting_time'] for v in self.vehicles[direction]['buses'])
                avg_waiting = total_waiting_sum / total_vehicles / 100.0  # Normalize by max expected waiting time ~100
            else:
                avg_waiting = 0.0
            state.append(avg_waiting)
    
        # Approaching vehicles near the intersection (4 directions), normalized by 10
        for direction in self.directions:
            approaching_cars = len([v for v in self.vehicles[direction]['cars'] if v['position'] > 0.5])
            approaching_buses = len([v for v in self.vehicles[direction]['buses'] if v['position'] > 0.5])
            approaching = (approaching_cars + approaching_buses) / 10.0
            state.append(min(approaching, 1.0))
    
        # Throughput indicator (total vehicles passed normalized)
        throughput_norm = min(self.total_vehicles_passed / 100.0, 1.0)
        state.append(throughput_norm)

        # Pad with zeros if length < 25 to avoid shape mismatch
        while len(state) < 25:
            state.append(0.0)

        # Truncate if accidentally longer than 25 (should not happen)
        state = state[:25]

        return np.array(state, dtype=np.float32)


    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Spawn vehicles
        self._spawn_vehicles()

        # Update phase based on action
        old_phase = self.current_phase
        if action == 0:  # North-South Green
            if self.current_phase != 0:
                self.current_phase = 0
                self.phase_timer = 0
        elif action == 1:  # East-West Green
            if self.current_phase != 1:
                self.current_phase = 1
                self.phase_timer = 0
        elif action == 2:  # All Red (emergency/transition)
            self.current_phase = 2
            self.phase_timer = 0

        # Update vehicles and calculate metrics
        old_total_waiting = self.total_waiting_time
        old_queue_lengths = self._get_queue_lengths()
        old_throughput = self.total_vehicles_passed

        self._update_vehicles()
        self.phase_timer += 1
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward(old_total_waiting, old_queue_lengths, old_throughput)

        # Episode ends after 1000 steps
        done = self.step_count >= 2000

        return self._get_state(), reward, done

    def _spawn_vehicles(self):
        """Spawn vehicles with different probabilities"""
        for direction in self.directions:
            # Spawn cars
            if random.random() < self.spawn_probability['car']:
                if len(self.vehicles[direction]['cars']) < self.max_cars_per_lane:
                    self.vehicles[direction]['cars'].append({
                        'position': 0.0,
                        'waiting_time': 0,
                        'speed': random.uniform(0.8, 1.0)
                    })

            # Spawn buses (less frequent, slower)
            if random.random() < self.spawn_probability['bus']:
                if len(self.vehicles[direction]['buses']) < self.max_cars_per_lane // 3:
                    self.vehicles[direction]['buses'].append({
                        'position': 0.0,
                        'waiting_time': 0,
                        'speed': random.uniform(0.5, 0.7)
                    })

    def _update_vehicles(self):
        """Update vehicle positions and waiting times"""
        for direction in self.directions:
            for vehicle_type in ['cars', 'buses']:
                vehicles_to_remove = []

                for i, vehicle in enumerate(self.vehicles[direction][vehicle_type]):
                    can_move = self._can_vehicle_move(direction, i, vehicle_type)

                    if can_move:
                        vehicle['position'] += vehicle['speed'] * 0.1
                        if vehicle['position'] >= 1.0:
                            vehicles_to_remove.append(i)
                            self.total_vehicles_passed += 1
                    else:
                        vehicle['waiting_time'] += 1
                        self.total_waiting_time += 1

                # Remove vehicles that passed through
                for i in reversed(vehicles_to_remove):
                    self.vehicles[direction][vehicle_type].pop(i)

    def _can_vehicle_move(self, direction, vehicle_index, vehicle_type):
        """Determine if vehicle can move based on traffic lights and congestion"""
        # Check traffic light
        if self.current_phase == 0 and direction in ['north', 'south']:
            light_allows = True
        elif self.current_phase == 1 and direction in ['east', 'west']:
            light_allows = True
        else:
            light_allows = False

        if not light_allows:
            return False

        # Check if there's a vehicle ahead
        vehicle = self.vehicles[direction][vehicle_type][vehicle_index]
        ahead_position = vehicle['position'] + 0.15  # Vehicle length

        # Check vehicles in front
        for other_type in ['cars', 'buses']:
            for other_vehicle in self.vehicles[direction][other_type]:
                if other_vehicle['position'] > vehicle['position'] and other_vehicle['position'] < ahead_position:
                    return False

        return True

    def _get_queue_lengths(self):
        """Get current queue lengths"""
        queue_lengths = []
        for direction in self.directions:
            cars = len(self.vehicles[direction]['cars'])
            buses = len(self.vehicles[direction]['buses'])
            queue_lengths.append(cars + buses * 2)  # Buses count as 2 cars
        return queue_lengths

    def _calculate_reward(self, old_waiting, old_queues, old_throughput):
        """Calculate reward combining waiting time, queue length, and throughput"""
        # Weights for different components (research-based optimal values)
        w_waiting = -0.4
        w_queue = -0.3
        w_throughput = 0.3

        # Waiting time component
        waiting_change = self.total_waiting_time - old_waiting
        waiting_reward = w_waiting * (waiting_change / 10.0)  # Normalize

        # Queue length component
        current_queues = self._get_queue_lengths()
        queue_penalty = w_queue * (sum(current_queues) / 40.0)  # Normalize

        # Throughput component
        throughput_change = self.total_vehicles_passed - old_throughput
        throughput_reward = w_throughput * throughput_change

        # Combine rewards
        total_reward = waiting_reward + queue_penalty + throughput_reward

        # Bonus for balanced queues
        if len(set(current_queues)) <= 2:  # Similar queue lengths
            total_reward += 0.1

        return total_reward

# Deep Q Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = lr
        self.gamma = 0.95
        self.batch_size = 32  # Optimized for GTX 1660 Ti

        # Neural Networks
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def save(self, filename):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# PyGame Visualization
class TrafficVisualization:
    def __init__(self, env, agent):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Adaptive Traffic Signal Control - Deep Q Learning")

        self.env = env
        self.agent = agent
        self.clock = pygame.time.Clock()

        # Colors
        self.colors = {
            'background': (50, 50, 50),
            'road': (70, 70, 70),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'blue': (0, 0, 255),
            'car': (0, 150, 255),
            'bus': (255, 100, 0),
            'text': (255, 255, 255)
        }

        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)

        # UI elements
        self.buttons = self._create_buttons()
        self.sliders = self._create_sliders()

        # Training metrics
        self.episode_rewards = []
        self.episode_waiting_times = []
        self.episode_queue_lengths = []

    def _create_buttons(self):
        """Create control buttons"""
        buttons = {}
        buttons['save'] = pygame.Rect(900, 50, 100, 40)
        buttons['restart'] = pygame.Rect(1020, 50, 100, 40)
        buttons['pause'] = pygame.Rect(900, 100, 220, 40)
        return buttons

    def _create_sliders(self):
        """Create parameter sliders"""
        sliders = {}
        sliders['epsilon'] = {
            'rect': pygame.Rect(900, 200, 200, 20),
            'value': self.agent.epsilon,
            'min': 0.0, 'max': 1.0,
            'label': 'Epsilon'
        }
        sliders['learning_rate'] = {
            'rect': pygame.Rect(900, 250, 200, 20),
            'value': self.agent.learning_rate,
            'min': 0.0001, 'max': 0.01,
            'label': 'Learning Rate'
        }
        return sliders

    def update_display(self, episode, step, reward, avg_waiting_time, avg_queue_length):
        """Update the visualization display"""
        self.screen.fill(self.colors['background'])

        # Draw intersection
        self._draw_intersection()

        # Draw vehicles
        self._draw_vehicles()

        # Draw traffic lights
        self._draw_traffic_lights()

        # Draw UI elements
        self._draw_ui(episode, step, reward, avg_waiting_time, avg_queue_length)

        # Draw metrics
        self._draw_metrics()

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def _draw_intersection(self):
        """Draw the 4-way intersection"""
        center_x, center_y = 400, 300
        road_width = 80

        # Horizontal road
        pygame.draw.rect(self.screen, self.colors['road'], 
                        (0, center_y - road_width//2, 800, road_width))

        # Vertical road
        pygame.draw.rect(self.screen, self.colors['road'], 
                        (center_x - road_width//2, 0, road_width, 600))

        # Lane dividers
        for i in range(-2, 3):
            if i != 0:
                # Horizontal dividers
                pygame.draw.line(self.screen, self.colors['white'],
                               (0, center_y + i * 20), (800, center_y + i * 20), 2)
                # Vertical dividers
                pygame.draw.line(self.screen, self.colors['white'],
                               (center_x + i * 20, 0), (center_x + i * 20, 600), 2)

    def _draw_vehicles(self):
        """Draw vehicles on the intersection"""
        center_x, center_y = 400, 300

        # Vehicle positions for each direction
        positions = {
            'north': (center_x - 10, lambda pos: center_y - 200 + pos * 180),
            'south': (center_x + 10, lambda pos: center_y + 200 - pos * 180),
            'east': (lambda pos: center_x + 200 - pos * 180, center_y - 10),
            'west': (lambda pos: center_x - 200 + pos * 180, center_y + 10)
        }

        for direction in self.env.directions:
            # Draw cars
            for i, vehicle in enumerate(self.env.vehicles[direction]['cars']):
                if direction in ['north', 'south']:
                    x = positions[direction][0]
                    y = positions[direction][1](vehicle['position'])
                    pygame.draw.rect(self.screen, self.colors['car'], (x-8, y-15, 16, 30))
                else:
                    x = positions[direction][0](vehicle['position'])
                    y = positions[direction][1]
                    pygame.draw.rect(self.screen, self.colors['car'], (x-15, y-8, 30, 16))

            # Draw buses
            for i, vehicle in enumerate(self.env.vehicles[direction]['buses']):
                if direction in ['north', 'south']:
                    x = positions[direction][0] + 25
                    y = positions[direction][1](vehicle['position'])
                    pygame.draw.rect(self.screen, self.colors['bus'], (x-10, y-20, 20, 40))
                else:
                    x = positions[direction][0](vehicle['position'])
                    y = positions[direction][1] + 25
                    pygame.draw.rect(self.screen, self.colors['bus'], (x-20, y-10, 40, 20))

    def _draw_traffic_lights(self):
        """Draw traffic lights with current states"""
        center_x, center_y = 400, 300
        light_positions = {
            'north': (center_x - 60, center_y - 60),
            'south': (center_x + 60, center_y + 60),
            'east': (center_x + 60, center_y - 60),
            'west': (center_x - 60, center_y + 60)
        }

        for direction, pos in light_positions.items():
            # Light background
            pygame.draw.rect(self.screen, (0, 0, 0), (pos[0]-15, pos[1]-30, 30, 60))

            # Determine light color
            if self.env.current_phase == 0 and direction in ['north', 'south']:
                color = self.colors['green']
            elif self.env.current_phase == 1 and direction in ['east', 'west']:
                color = self.colors['green']
            else:
                color = self.colors['red']

            # Draw light
            pygame.draw.circle(self.screen, color, pos, 12)

            # Phase timer
            timer_text = self.font.render(str(self.env.phase_timer), True, self.colors['white'])
            self.screen.blit(timer_text, (pos[0]-10, pos[1]+20))

    def _draw_ui(self, episode, step, reward, avg_waiting_time, avg_queue_length):
        """Draw UI elements and metrics"""
        # Episode info
        episode_text = self.large_font.render(f"Episode: {episode}", True, self.colors['text'])
        self.screen.blit(episode_text, (900, 10))

        # Metrics
        metrics = [
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Avg Waiting: {avg_waiting_time:.1f}",
            f"Avg Queue: {avg_queue_length:.1f}",
            f"Phase: {['NS Green', 'EW Green', 'All Red'][self.env.current_phase]}",
            f"Phase Timer: {self.env.phase_timer}"
        ]

        for i, metric in enumerate(metrics):
            text = self.font.render(metric, True, self.colors['text'])
            self.screen.blit(text, (900, 300 + i * 25))

        # Agent parameters
        agent_info = [
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Memory: {len(self.agent.memory)}/{self.agent.memory.maxlen}",
            f"Learning Rate: {self.agent.learning_rate:.4f}"
        ]

        for i, info in enumerate(agent_info):
            text = self.font.render(info, True, self.colors['text'])
            self.screen.blit(text, (900, 500 + i * 25))

        # Draw buttons
        for name, rect in self.buttons.items():
            pygame.draw.rect(self.screen, self.colors['blue'], rect)
            button_text = self.font.render(name.title(), True, self.colors['white'])
            text_rect = button_text.get_rect(center=rect.center)
            self.screen.blit(button_text, text_rect)

        # Draw sliders
        for name, slider in self.sliders.items():
            pygame.draw.rect(self.screen, self.colors['white'], slider['rect'], 2)
            # Slider handle
            handle_x = slider['rect'].x + (slider['value'] - slider['min']) / (slider['max'] - slider['min']) * slider['rect'].width
            pygame.draw.circle(self.screen, self.colors['blue'], (int(handle_x), slider['rect'].centery), 8)
            # Label
            label_text = self.font.render(f"{slider['label']}: {slider['value']:.3f}", True, self.colors['text'])
            self.screen.blit(label_text, (slider['rect'].x, slider['rect'].y - 25))

    def _draw_metrics(self):
        """Draw training progress metrics"""
        if len(self.episode_rewards) > 1:
            # Simple line graph for rewards
            graph_rect = pygame.Rect(900, 600, 280, 120)
            pygame.draw.rect(self.screen, (30, 30, 30), graph_rect)
            pygame.draw.rect(self.screen, self.colors['white'], graph_rect, 2)

            # Plot last 50 episodes
            rewards_to_plot = self.episode_rewards[-50:]
            if len(rewards_to_plot) > 1:
                max_reward = max(rewards_to_plot)
                min_reward = min(rewards_to_plot)
                range_reward = max_reward - min_reward if max_reward != min_reward else 1

                points = []
                for i, reward in enumerate(rewards_to_plot):
                    x = graph_rect.x + (i / (len(rewards_to_plot) - 1)) * graph_rect.width
                    y = graph_rect.bottom - ((reward - min_reward) / range_reward) * graph_rect.height
                    points.append((x, y))

                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.colors['green'], False, points, 2)

        # Graph title
        title = self.font.render("Episode Rewards", True, self.colors['text'])
        self.screen.blit(title, (900, 580))

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                # Check button clicks
                for name, rect in self.buttons.items():
                    if rect.collidepoint(mouse_pos):
                        if name == 'save':
                            filename = f"traffic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                            self.agent.save(filename)
                            print(f"Model saved as {filename}")
                        elif name == 'restart':
                            self.env.reset()
                            print("Environment reset")
                        elif name == 'pause':
                            print("Training paused - press any key to continue")
                            pygame.event.wait()

                # Check slider interactions
                for name, slider in self.sliders.items():
                    if slider['rect'].collidepoint(mouse_pos):
                        relative_x = mouse_pos[0] - slider['rect'].x
                        slider_value = slider['min'] + (relative_x / slider['rect'].width) * (slider['max'] - slider['min'])
                        slider_value = max(slider['min'], min(slider['max'], slider_value))
                        slider['value'] = slider_value

                        # Update agent parameters
                        if name == 'epsilon':
                            self.agent.epsilon = slider_value
                        elif name == 'learning_rate':
                            self.agent.learning_rate = slider_value
                            for param_group in self.agent.optimizer.param_groups:
                                param_group['lr'] = slider_value

        return True

# Training Function
def train_agent():
    """Main training function"""
    env = TrafficEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    visualization = TrafficVisualization(env, agent)

    episodes = 2000
    target_update_frequency = 100
    save_frequency = 50

    print(f"Starting training for {episodes} episodes")
    print(f"Device: {device}")
    print(f"State size: {env.state_size}, Action size: {env.action_size}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            # Handle pygame events
            if not visualization.handle_events():
                return  # User closed window
        
            # Select action
            action = agent.act(state)
        
            # Take step in environment
            next_state, reward, done = env.step(action)
        
            # Store experience
            agent.remember(state, action, reward, next_state, done)
        
            # Train on batch
            agent.replay()
        
            # Update metrics
            total_reward += reward
        
            # Update visualization periodically
            if step % 5 == 0:
                avg_waiting = env.total_waiting_time / (step + 1)
                avg_queue = sum(env._get_queue_lengths()) / (step + 1)
                visualization.update_display(episode, step, total_reward, avg_waiting, avg_queue)
        
            state = next_state
            step += 1

        # Decay epsilon once per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={step}, Epsilon={agent.epsilon:.3f}")

    # Save final model
    agent.save("traffic_model_final.pth")
    print("Training completed! Final model saved.")

    # Keep window open until closed
    while visualization.handle_events():
        visualization.update_display(episodes, 0, 0, 0, 0)

    pygame.quit()

if __name__ == "__main__":
    train_agent()
