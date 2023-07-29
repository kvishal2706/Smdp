import random
from scipy.optimize import linprog

class SMDP:
    def __init__(self, state_space, action_space, gamma=0.9, Es=10, Et=10, Er=10):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.Es = Es
        self.Et = Et
        self.Er = Er
        self.transition_probabilities = {}  # Dictionary to store state transition probabilities

        # Initialize transition probabilities for each state-action pair
        for state in state_space:
            self.transition_probabilities[state] = {}
            for action in action_space:
                self.transition_probabilities[state][action] = {}

    def add_transition_prob(self, state, action, next_state, probability):
        # Add a state transition probability for a specific state-action pair
        self.transition_probabilities[state][action][next_state] = probability

    def get_reward(self, state, action, next_state):
        x1, e_s = state
        x1_next, e_s_next = next_state

        # Implement the reward calculation based on the research paper's equations
        if e_s == "a1" and e_s_next == "a1" and action == "a3":
            reward = -self.Er  # Penalty for rejecting a service request
        elif e_s == "a1" and e_s_next == "a1" and action == "a1":
            reward = self.Es - self.Et  # Income from using security VMs minus penalty for using normal VMs
        else:
            reward = 0  # Default reward if the above conditions are not met

        return reward

    def calculate_optimal_policy(self):
        # Define the coefficients of the objective function
        c = [-1] * (len(self.state_space) * len(self.action_space))

        # Define the inequality constraints (Ax <= b)
        A = []
        b = []

        for state in self.state_space:
            for action in self.action_space:
                # Define the constraint: Sum of probabilities of next states for each action = 1
                constraint = [0] * (len(self.state_space) * len(self.action_space))
                for next_state in self.state_space:
                    constraint[self.state_space.index(next_state) + len(self.state_space) * self.action_space.index(action)] = -self.transition_probabilities[state][action].get(next_state, 0)
                A.append(constraint)
                b.append(-1)

        # Define the equality constraint: Probability of being in a state after taking an action = 1
        for state in self.state_space:
            constraint = [0] * (len(self.state_space) * len(self.action_space))
            for action in self.action_space:
                constraint[self.state_space.index(state) + len(self.state_space) * len(self.action_space) * action] = 1
            A.append(constraint)
            b.append(1)

        # Bounds for probabilities (0 <= P(state' | state, action) <= 1)
        bounds = [(0, 1)] * (len(self.state_space) * len(self.action_space))

        # Solve the LP problem
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

        # Extract the optimal policy from the result
        optimal_policy = {}
        for state in self.state_space:
            state_probs = result.x[self.state_space.index(state)::len(self.state_space)]
            optimal_policy[state] = {action: state_probs[self.action_space.index(action)] for action in self.action_space}

        return optimal_policy

    def reset(self):
        self.current_state = random.choice(self.state_space)

    def step(self, action):
        if self.current_state is None:
            raise ValueError("You must call 'reset()' before starting the episode.")

        transition_probs = self.transition_probabilities[self.current_state][action]
        next_state = random.choices(list(transition_probs.keys()), weights=list(transition_probs.values()))[0]
        reward = self.get_reward(self.current_state, action, next_state)
        done = False  # Replace with termination condition based on your problem

        # Move to the next state after checking the termination condition
        self.current_state = next_state

        return next_state, reward, done


if __name__ == "__main__":
    state_space = [(x1, e_s) for x1 in range(17) for e_s in ["a1", "a2", "a3"]]
    action_space = ["a1", "a2", "a3"]

    smdp = SMDP(state_space, action_space)

    # Add state transition probabilities based on Table III and Table IV in the research paper
    # Table III
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a1"), 1.0)
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a2"), 1.0)
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a3"), 1.0)
    # Add other transition probabilities based on Table III and Table IV
    # Table IV
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a1"), 1.0)
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a2"), 1.0)
    smdp.add_transition_prob((0, "a3"), "a3", (3, "a3"), 1.0)

    for _ in range(5):
        smdp.reset()
        total_reward = 0
        done = False

        # Calculate the optimal policy
        optimal_policy = smdp.calculate_optimal_policy()

        while not done:
            # Implement the optimal policy to select the best action based on the current state
            action_probs = optimal_policy[smdp.current_state]
            action = max(action_probs, key=action_probs.get)  # Choose the action with the highest probability
            next_state, reward, done = smdp.step(action)
            total_reward += reward
            smdp.current_state = next_state

        print(f"Total reward for episode: {total_reward}")
