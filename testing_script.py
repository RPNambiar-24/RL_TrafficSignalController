
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
import sys
import os
import json
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Import classes from training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class TrafficEnvironment:
    def __init__(self):
        
        self.max_cars_per_lane = 20
        self.spawn_probability = {'car': 0.3, 'bus': 0.1}
        self.max_green_time = 60
        self.min_green_time = 10
        self.yellow_time = 3
        self.all_red_time = 2
        self.state_size = 25
        self.action_size = 4
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

        self.current_phase = 0
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
        self._spawn_vehicles()

        old_phase = self.current_phase
        if action == 0:
            if self.current_phase != 0:
                self.current_phase = 0
                self.phase_timer = 0
        elif action == 1:
            if self.current_phase != 1:
                self.current_phase = 1
                self.phase_timer = 0
        elif action == 2:
            self.current_phase = 2
            self.phase_timer = 0

        old_total_waiting = self.total_waiting_time
        old_queue_lengths = self._get_queue_lengths()
        old_throughput = self.total_vehicles_passed

        self._update_vehicles()
        self.phase_timer += 1
        self.step_count += 1

        reward = self._calculate_reward(old_total_waiting, old_queue_lengths, old_throughput)
        done = self.step_count >= 2000

        return self._get_state(), reward, done

    def _spawn_vehicles(self):
        """Spawn vehicles with different probabilities"""
        import random
        for direction in self.directions:
            if random.random() < self.spawn_probability['car']:
                if len(self.vehicles[direction]['cars']) < self.max_cars_per_lane:
                    self.vehicles[direction]['cars'].append({
                        'position': 0.0,
                        'waiting_time': 0,
                        'speed': random.uniform(0.8, 1.0)
                    })

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

                for i in reversed(vehicles_to_remove):
                    self.vehicles[direction][vehicle_type].pop(i)

    def _can_vehicle_move(self, direction, vehicle_index, vehicle_type):
        """Determine if vehicle can move based on traffic lights and congestion"""
        if self.current_phase == 0 and direction in ['north', 'south']:
            light_allows = True
        elif self.current_phase == 1 and direction in ['east', 'west']:
            light_allows = True
        else:
            light_allows = False

        if not light_allows:
            return False

        vehicle = self.vehicles[direction][vehicle_type][vehicle_index]
        ahead_position = vehicle['position'] + 0.15

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
            queue_lengths.append(cars + buses * 2)
        return queue_lengths

    def _calculate_reward(self, old_waiting, old_queues, old_throughput):
        """Calculate reward combining waiting time, queue length, and throughput"""
        w_waiting = -0.4
        w_queue = -0.3
        w_throughput = 0.3

        waiting_change = self.total_waiting_time - old_waiting
        waiting_reward = w_waiting * (waiting_change / 10.0)

        current_queues = self._get_queue_lengths()
        queue_penalty = w_queue * (sum(current_queues) / 40.0)

        throughput_change = self.total_vehicles_passed - old_throughput
        throughput_reward = w_throughput * throughput_change

        total_reward = waiting_reward + queue_penalty + throughput_reward

        if len(set(current_queues)) <= 2:
            total_reward += 0.1

        return total_reward

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = DQN(state_size, action_size).to(device)

    def act(self, state, epsilon=0.0):
        """Choose action (greedy for testing)"""
        if np.random.random() <= epsilon:
            return np.random.choice(range(self.action_size))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def load(self, filename):
        """Load model"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"Model file {filename} not found!")
            return False

class TestVisualization:
    def __init__(self, env, agent):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Traffic Signal Control - Testing Mode")

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

        # Testing metrics
        self.test_metrics = {
            'total_vehicles': 0,
            'total_waiting_time': 0,
            'total_steps': 0,
            'episodes_completed': 0,
            'average_reward': 0,
            'average_queue_length': 0
        }

        # Create control buttons
        self.buttons = {
            'reset': pygame.Rect(900, 50, 100, 40),
            'new_test': pygame.Rect(1020, 50, 100, 40),
            'save_results': pygame.Rect(900, 100, 220, 40)
        }

    def update_display(self, episode, step, reward, performance_metrics):
        """Update the testing visualization"""
        self.screen.fill(self.colors['background'])

        # Draw intersection
        self._draw_intersection()

        # Draw vehicles
        self._draw_vehicles()

        # Draw traffic lights
        self._draw_traffic_lights()

        # Draw testing UI
        self._draw_test_ui(episode, step, reward, performance_metrics)

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_intersection(self):
        """Draw the 4-way intersection"""
        center_x, center_y = 400, 300
        road_width = 80

        pygame.draw.rect(self.screen, self.colors['road'], 
                        (0, center_y - road_width//2, 800, road_width))
        pygame.draw.rect(self.screen, self.colors['road'], 
                        (center_x - road_width//2, 0, road_width, 600))

        for i in range(-2, 3):
            if i != 0:
                pygame.draw.line(self.screen, self.colors['white'],
                               (0, center_y + i * 20), (800, center_y + i * 20), 2)
                pygame.draw.line(self.screen, self.colors['white'],
                               (center_x + i * 20, 0), (center_x + i * 20, 600), 2)

    def _draw_vehicles(self):
        """Draw vehicles on the intersection"""
        center_x, center_y = 400, 300

        positions = {
            'north': (center_x - 10, lambda pos: center_y - 200 + pos * 180),
            'south': (center_x + 10, lambda pos: center_y + 200 - pos * 180),
            'east': (lambda pos: center_x + 200 - pos * 180, center_y - 10),
            'west': (lambda pos: center_x - 200 + pos * 180, center_y + 10)
        }

        for direction in self.env.directions:
            for i, vehicle in enumerate(self.env.vehicles[direction]['cars']):
                if direction in ['north', 'south']:
                    x = positions[direction][0]
                    y = positions[direction][1](vehicle['position'])
                    pygame.draw.rect(self.screen, self.colors['car'], (x-8, y-15, 16, 30))
                else:
                    x = positions[direction][0](vehicle['position'])
                    y = positions[direction][1]
                    pygame.draw.rect(self.screen, self.colors['car'], (x-15, y-8, 30, 16))

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
            pygame.draw.rect(self.screen, (0, 0, 0), (pos[0]-15, pos[1]-30, 30, 60))

            if self.env.current_phase == 0 and direction in ['north', 'south']:
                color = self.colors['green']
            elif self.env.current_phase == 1 and direction in ['east', 'west']:
                color = self.colors['green']
            else:
                color = self.colors['red']

            pygame.draw.circle(self.screen, color, pos, 12)

            timer_text = self.font.render(str(self.env.phase_timer), True, self.colors['white'])
            self.screen.blit(timer_text, (pos[0]-10, pos[1]+20))

    def _draw_test_ui(self, episode, step, reward, metrics):
        """Draw testing UI and metrics"""
        # Title
        title_text = self.large_font.render("TESTING MODE", True, self.colors['text'])
        self.screen.blit(title_text, (900, 10))

        # Current episode info
        current_info = [
            f"Test Episode: {episode}",
            f"Step: {step}",
            f"Current Reward: {reward:.2f}",
            f"Phase: {['NS Green', 'EW Green', 'All Red'][self.env.current_phase]}",
            f"Phase Timer: {self.env.phase_timer}"
        ]

        for i, info in enumerate(current_info):
            text = self.font.render(info, True, self.colors['text'])
            self.screen.blit(text, (900, 200 + i * 25))

        # Performance metrics
        perf_title = self.font.render("Performance Metrics:", True, self.colors['white'])
        self.screen.blit(perf_title, (900, 350))

        perf_metrics = [
            f"Avg Waiting Time: {metrics.get('avg_waiting', 0):.1f}",
            f"Avg Queue Length: {metrics.get('avg_queue', 0):.1f}",
            f"Total Vehicles: {metrics.get('total_vehicles', 0)}",
            f"Throughput: {metrics.get('throughput', 0):.2f} v/min",
            f"Episodes Tested: {self.test_metrics['episodes_completed']}",
            f"Overall Avg Reward: {self.test_metrics['average_reward']:.2f}"
        ]

        for i, metric in enumerate(perf_metrics):
            text = self.font.render(metric, True, self.colors['text'])
            self.screen.blit(text, (900, 380 + i * 25))

        # Queue lengths by direction
        queue_title = self.font.render("Queue Lengths:", True, self.colors['white'])
        self.screen.blit(queue_title, (900, 550))

        queue_lengths = self.env._get_queue_lengths()
        for i, (direction, queue) in enumerate(zip(self.env.directions, queue_lengths)):
            text = self.font.render(f"{direction.title()}: {queue}", True, self.colors['text'])
            self.screen.blit(text, (900, 580 + i * 20))

        # Draw buttons
        for name, rect in self.buttons.items():
            pygame.draw.rect(self.screen, self.colors['blue'], rect)
            button_text = self.font.render(name.replace('_', ' ').title(), True, self.colors['white'])
            text_rect = button_text.get_rect(center=rect.center)
            self.screen.blit(button_text, text_rect)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                for name, rect in self.buttons.items():
                    if rect.collidepoint(mouse_pos):
                        if name == 'reset':
                            self.env.reset()
                            print("Environment reset for new test")
                        elif name == 'new_test':
                            self._start_new_test()
                        elif name == 'save_results':
                            self._save_test_results()

        return True

    def _start_new_test(self):
        """Start a new test episode"""
        self.env.reset()
        self.test_metrics['episodes_completed'] += 1
        print(f"Starting new test episode {self.test_metrics['episodes_completed']}")

    def _save_test_results(self):
        """Save test results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'episodes_tested': self.test_metrics['episodes_completed'],
            'average_reward': self.test_metrics['average_reward'],
            'average_waiting_time': self.test_metrics['total_waiting_time'] / max(1, self.test_metrics['total_steps']),
            'average_queue_length': self.test_metrics['average_queue_length'],
            'total_vehicles_processed': self.test_metrics['total_vehicles']
        }

        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Test results saved to {filename}")

def test_model(model_filename):
    """Main testing function"""
    print(f"Loading model: {model_filename}")

    env = TrafficEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)

    # Load trained model
    if not agent.load(model_filename):
        print("Failed to load model. Exiting...")
        return

    visualization = TestVisualization(env, agent)

    print("Testing mode started. Controls:")
    print("- Reset: Reset current episode")
    print("- New Test: Start new test episode")
    print("- Save Results: Save performance metrics")
    print("- Close window to exit")

    test_episodes = 10
    episode_rewards = []

    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        episode_waiting_time = 0
        episode_queue_length = 0

        print(f"\nStarting test episode {episode + 1}/{test_episodes}")

        while True:
            if not visualization.handle_events():
                return

            # Choose action (greedy - no exploration)
            action = agent.act(state, epsilon=0.0)

            # Take step
            next_state, reward, done = env.step(action)

            # Update metrics
            total_reward += reward
            episode_waiting_time += env.total_waiting_time
            episode_queue_length += sum(env._get_queue_lengths())

            # Performance metrics for display
            avg_waiting = episode_waiting_time / (step + 1) if step > 0 else 0
            avg_queue = episode_queue_length / (step + 1) if step > 0 else 0
            throughput = env.total_vehicles_passed / max(1, (step + 1) / 60.0)  # vehicles per minute

            performance_metrics = {
                'avg_waiting': avg_waiting,
                'avg_queue': avg_queue,
                'total_vehicles': env.total_vehicles_passed,
                'throughput': throughput
            }

            # Update visualization
            visualization.update_display(episode + 1, step, total_reward, performance_metrics)

            state = next_state
            step += 1

            if done:
                break

        # Update overall test metrics
        episode_rewards.append(total_reward)
        visualization.test_metrics['total_vehicles'] += env.total_vehicles_passed
        visualization.test_metrics['total_waiting_time'] += episode_waiting_time
        visualization.test_metrics['total_steps'] += step
        visualization.test_metrics['episodes_completed'] = episode + 1
        visualization.test_metrics['average_reward'] = sum(episode_rewards) / len(episode_rewards)
        visualization.test_metrics['average_queue_length'] = episode_queue_length / step

        print(f"Episode {episode + 1} completed:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Vehicles Passed: {env.total_vehicles_passed}")
        print(f"  Avg Waiting Time: {avg_waiting:.1f}")
        print(f"  Avg Queue Length: {avg_queue:.1f}")

    # Final results
    print(f"\n=== TESTING COMPLETED ===")
    print(f"Episodes tested: {test_episodes}")
    print(f"Average reward: {visualization.test_metrics['average_reward']:.2f}")
    print(f"Total vehicles processed: {visualization.test_metrics['total_vehicles']}")
    print(f"Overall average waiting time: {visualization.test_metrics['total_waiting_time'] / visualization.test_metrics['total_steps']:.1f}")

    # Keep window open for further analysis
    print("\nWindow will remain open for analysis. Close to exit.")
    while visualization.handle_events():
        # Show final results
        final_metrics = {
            'avg_waiting': visualization.test_metrics['total_waiting_time'] / visualization.test_metrics['total_steps'],
            'avg_queue': visualization.test_metrics['average_queue_length'],
            'total_vehicles': visualization.test_metrics['total_vehicles'],
            'throughput': visualization.test_metrics['total_vehicles'] / (visualization.test_metrics['total_steps'] / 60.0)
        }
        visualization.update_display(test_episodes, 0, visualization.test_metrics['average_reward'], final_metrics)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testing_script.py <model_filename>")
        print("Example: python testing_script.py traffic_model_final.pth")
        sys.exit(1)

    model_file = sys.argv[1]
    test_model(model_file)
