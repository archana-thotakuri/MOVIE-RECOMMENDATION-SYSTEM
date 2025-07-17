import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class SmartIrrigationSystem:
    def _init_(self, root):
        self.root = root
        self.root.title("IoT Based Smart Irrigation System using Reinforcement Learning")
        self.root.geometry("1000x600")
        self.root.configure(bg="lightcoral")

        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.q_table = None

        # Buttons
        tk.Button(root, text="Upload Irrigation Dataset", command=self.upload_dataset).pack(pady=10)
        tk.Button(root, text="Preprocess Dataset", command=self.preprocess_dataset).pack(pady=10)
        tk.Button(root, text="Dataset Train & Test Split", command=self.train_test_split_dataset).pack(pady=10)
        tk.Button(root, text="Train Reinforcement Learning Algorithm", command=self.train_rl_algorithm).pack(pady=10)
        tk.Button(root, text="Rewards & Penalty Graph", command=self.plot_rewards_penalty_graph).pack(pady=10)
        tk.Button(root, text="Predict Irrigation Status", command=self.predict_status).pack(pady=10)
        tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.dataset = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Dataset uploaded successfully!")
            print(self.dataset.head())

    def preprocess_dataset(self):
        if self.dataset is not None:
            # Show distribution of irrigation labels
            plt.figure(figsize=(6, 4))
            self.dataset['class'].value_counts().plot(kind='bar')
            plt.xlabel('Irrigation Condition Labels')
            plt.ylabel('Count')
            plt.title('Distribution of Irrigation Conditions')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Please upload the dataset first!")

    def train_test_split_dataset(self):
        if self.dataset is not None:
            self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
            messagebox.showinfo("Success", "Dataset split into Train and Test sets.")
        else:
            messagebox.showwarning("Warning", "Please upload the dataset first!")

    def train_rl_algorithm(self):
        if self.train_data is not None:
            states = self.train_data['class'].unique()
            actions = ['No Irrigation', 'Light Irrigation', 'Heavy Irrigation']

            self.q_table = pd.DataFrame(0, index=states, columns=actions)

            learning_rate = 0.1
            discount_factor = 0.9
            episodes = 1000

            for _ in range(episodes):
                state = np.random.choice(states)
                action = np.random.choice(actions)

                reward = self.get_reward(state, action)

                old_value = self.q_table.loc[state, action]
                next_max = self.q_table.loc[state].max()

                # Q-learning formula
                new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
                self.q_table.loc[state, action] = new_value

            messagebox.showinfo("Success", "Reinforcement Learning Model Trained!")
            print(self.q_table)
        else:
            messagebox.showwarning("Warning", "Please preprocess and split the dataset first!")

    def get_reward(self, state, action):
        # Custom reward logic
        rewards = {
            ('Very Dry', 'Heavy Irrigation'): 10,
            ('Dry', 'Light Irrigation'): 8,
            ('Wet', 'No Irrigation'): 10,
            ('Very Wet', 'No Irrigation'): 10,
            ('Very Dry', 'No Irrigation'): -10,
            ('Wet', 'Heavy Irrigation'): -8,
            ('Very Wet', 'Heavy Irrigation'): -10
        }
        return rewards.get((state, action), 0)

    def plot_rewards_penalty_graph(self):
        if self.q_table is not None:
            self.q_table.plot(kind='bar', figsize=(10, 6))
            plt.title('Q-Table Rewards for Different Actions')
            plt.xlabel('Irrigation Condition')
            plt.ylabel('Q-Values')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Please train the model first!")

    def predict_status(self):
        if self.q_table is not None:
            sample_state = filedialog.askopenfilename(title="Select a sample state CSV file")
            if sample_state:
                sample = pd.read_csv(sample_state)
                irrigation_class = sample.iloc[0]['class']
                best_action = self.q_table.loc[irrigation_class].idxmax()
                messagebox.showinfo("Prediction", f"For {irrigation_class} condition, Recommended Action: {best_action}")
        else:
            messagebox.showwarning("Warning", "Please train the model first!")

if _name_ == "_main_":
    root = tk.Tk()
    app = SmartIrrigationSystem(root)
    root.mainloop()