import math
import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict

class WindyGridWorldState:    
    def __init__(self):
        # Grid configuration
        self.width = 10
        self.height = 7
        self.start = (0, 3)  # (x, y)
        self.goal = (7, 3)   # (x, y)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Wind strength for each column
        
        # Action mappings
        self.ACTIONS = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        
        # Current state
        self.x, self.y = self.start
        self.steps = 0
        self.max_steps = 100
        self.last_action = None

    def is_terminal(self) -> bool:
        return (self.x, self.y) == self.goal or self.steps >= self.max_steps

    def get_legal_actions(self) -> List[int]:
        return [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

    def apply_action(self, action: int) -> bool:
        # Apply action first
        if action == 0:    # UP
            self.y += 1
        elif action == 1:  # DOWN
            self.y -= 1
        elif action == 2:  # LEFT
            self.x -= 1
        elif action == 3:  # RIGHT
            self.x += 1

        # Ensure within boundaries after action
        self.x = min(max(self.x, 0), self.width - 1)
        self.y = min(max(self.y, 0), self.height - 1)

        # Then apply wind effect
        if 0 <= self.x < len(self.wind):
            wind_strength = self.wind[self.x]
            self.y += wind_strength
            # Ensure within boundaries after wind
            self.y = min(max(self.y, 0), self.height - 1)

        self.steps += 1
        self.last_action = action
        return True

    def get_reward(self) -> float:
        max_distance = self.width + self.height
        distance = abs(self.x - self.goal[0]) + abs(self.y - self.goal[1])
        return 1.0 - (distance / max_distance)

    def clone(self) -> 'WindyGridWorldState':
        new_state = WindyGridWorldState()
        new_state.x = self.x
        new_state.y = self.y
        new_state.steps = self.steps
        new_state.last_action = self.last_action
        return new_state

    def __str__(self) -> str:
        grid = []
        for _ in range(self.height):
            grid.append(['.' for _ in range(self.width)])
        
        grid[self.y][self.x] = 'P'
        grid[self.goal[1]][self.goal[0]] = 'G'
        
        wind_indicator = "Wind: " + " ".join(str(w) for w in self.wind)
        
        grid_rows = []
        for row in reversed(grid):
            grid_rows.append(" ".join(row))
        grid_str = "\n".join(grid_rows)
        
        return f"\nGrid (P=Player, G=Goal):\n{grid_str}\n{wind_indicator}"

@dataclass
class Node:
    state: WindyGridWorldState
    parent: Optional['Node'] = None
    children: List['Node'] = None
    wins: float = 0.0
    visits: int = 0
    untried_actions: List[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = self.state.get_legal_actions()

    def ucb1(self, exploration_constant: float = 0.5) -> float:
        if self.visits == 0 or self.parent is None or self.parent.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

class PolicyMCTS:
    def __init__(self, root_state: WindyGridWorldState, simulation_limit: int = 10):
        self.root = Node(state=root_state)
        self.simulation_limit = simulation_limit
        self.policy_table = defaultdict(lambda: defaultdict(lambda: (0, 0.0)))

    def get_policy_action(self, state_key: Tuple[int, int]) -> Optional[int]:
        actions = self.policy_table[state_key]
        if not actions:
            return None

        best_action = max(actions.items(),
                         key=lambda x: x[1][1]/x[1][0] if x[1][0] > 0 else float('-inf'))
        return best_action[0]

    def update_policy(self, trajectory: List[Tuple[Tuple[int, int], int]], reward: float):
        for (state_key, action) in trajectory:
            visits, wins = self.policy_table[state_key][action]
            self.policy_table[state_key][action] = (
                visits + 1,
                wins + reward
            )

    def search(self) -> Tuple[int, List[Tuple[Tuple[int, int], int]]]:
        for _ in range(self.simulation_limit):
            node = self.select(self.root)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        best_child = max(self.root.children,
                        key=lambda c: (c.wins / c.visits) if c.visits > 0 else float('-inf'))
        return best_child.state.last_action

    def select(self, node: Node) -> Node:
        current = node
        while not current.is_terminal():
            if not current.is_fully_expanded():
                return current
            current = max(current.children, key=lambda n: n.ucb1())
        return current

    def expand(self, node: Node) -> Node:
        if node.is_terminal() or not node.untried_actions:
            return node
            
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        new_state = node.state.clone()
        new_state.apply_action(action)
        
        child = Node(state=new_state, parent=node)
        node.children.append(child)
        return child

    def select_heuristic_action(self, state: WindyGridWorldState) -> int:
        actions = state.get_legal_actions()
        best_action = None
        min_distance = float('inf')
        for action in actions:
            temp_state = state.clone()
            temp_state.apply_action(action)
            distance = abs(temp_state.x - state.goal[0]) + abs(temp_state.y - state.goal[1])
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action

    def simulate(self, node: Node) -> float:
        state = node.state.clone()
        trajectory = []
        
        while not state.is_terminal():
            state_key = (state.x, state.y)
            policy_action = self.get_policy_action(state_key)
            
            if policy_action is not None:
                action = policy_action
            else:
                action = self.select_heuristic_action(state)
            
            trajectory.append((state_key, action))
            state.apply_action(action)
        
        reward = state.get_reward()
        self.update_policy(trajectory, reward)
        
        return reward

    def backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

def play_and_learn():
    episodes = 20
    best_steps = float('inf')
    global_policy_table = defaultdict(lambda: defaultdict(lambda: (0, 0.0)))
    best_trajectory = None
    
    print("Starting learning process...")
    
    for episode in range(episodes):
        state = WindyGridWorldState()
        trajectory = []
        
        while not state.is_terminal():
            state_key = (state.x, state.y)
            mcts = PolicyMCTS(state.clone(), simulation_limit=200)
            
            # Update MCTS policy table with global knowledge
            mcts.policy_table.update(global_policy_table)
            
            action = mcts.search()
            trajectory.append((state_key, action))
            state.apply_action(action)
            
            # Update global policy table
            global_policy_table.update(mcts.policy_table)

        if (state.x, state.y) == state.goal:
            print(f"Episode {episode + 1}: Success! Steps: {state.steps}")
            if state.steps < best_steps:
                best_steps = state.steps
                best_trajectory = trajectory
                print(f"New best path found! Length: {best_steps}")
        else:
            print(f"Episode {episode + 1}: Failed after {state.steps} steps")

    if best_trajectory:
        print("\nBest trajectory found:")
        state = WindyGridWorldState()
        print(state)
        
        for i, (_, action) in enumerate(best_trajectory, 1):
            print(f"Step {i}: {state.ACTIONS[action]}")
            state.apply_action(action)
            print(state)
        
        print(f"\nTotal steps: {best_steps}")
    else:
        print("\nNo successful trajectory found")

if __name__ == "__main__":
    play_and_learn()
