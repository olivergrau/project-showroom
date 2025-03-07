import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)    
except RuntimeError:
    # start method already set
    pass

import torch

import sys
import os

# Insert the parent directory into the system path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import time
import numpy as np

# Import the classes from your replay buffer module
from codebase.replay.replay_buffer import UniformReplay, Transition, ReplayWrapper
from codebase.replay.replay_proxy import ReplayProxy

class TestUniformReplay(unittest.TestCase):
    def setUp(self):
        # Define parameters for the replay buffer.
        self.memory_size = 10
        self.batch_size = 4
        self.keys = ['state', 'action', 'reward', 'mask', 'next_state']
        self.replay = UniformReplay(
            memory_size=self.memory_size, 
            batch_size=self.batch_size, 
            n_step=1, 
            discount=0.99, 
            history_length=1, 
            keys=self.keys
        )
        # Feed 15 dummy transitions. With memory_size 10, only the last 10 transitions are stored.
        for i in range(15):
            data = {
                'state': [np.array([i], dtype=np.float32)],
                'action': [np.array([i + 0.1], dtype=np.float32)],
                'reward': [float(i)],
                'mask': [1],
                'next_state': [np.array([i + 1], dtype=np.float32)]
            }
            self.replay.feed(data)

    def test_sample_shape_and_values(self):
        # Sample one batch.
        transition = self.replay.sample()
        # Check that each field is a numpy array with first dimension equal to batch_size.
        self.assertEqual(transition.state.shape[0], self.batch_size)
        self.assertEqual(transition.action.shape[0], self.batch_size)
        self.assertEqual(transition.reward.shape[0], self.batch_size)
        self.assertEqual(transition.mask.shape[0], self.batch_size)
        self.assertEqual(transition.next_state.shape[0], self.batch_size)
        
        # Since the buffer is circular with capacity 10 and we fed 15 transitions, stored indices come from transitions 5 to 14.
        for r in transition.reward:
            self.assertTrue(5 <= r <= 14)

class TestReplayWrapper(unittest.TestCase):
    def setUp(self):
        self.memory_size = 10
        self.batch_size = 4
        self.keys = ['state', 'action', 'reward', 'mask', 'next_state']
        self.replay_kwargs = {
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "n_step": 1,
            "discount": 0.99,
            "history_length": 1,
            "keys": self.keys
        }

        # Create the ReplayWrapper in asynchronous mode.
        self.wrapper = ReplayWrapper(UniformReplay, self.replay_kwargs, asynchronous=True)

        # Wait for a "ready" signal from the worker.
        ready, _ = self.wrapper.pipe.recv()
        self.assertEqual(ready, "ready")

        print(f"ReplayWrapper ready: {ready}")

        # Feed 15 dummy transitions.
        for i in range(15):
            data = {
                'state': [np.array([i], dtype=np.float32)],
                'action': [np.array([i + 0.1], dtype=np.float32)],
                'reward': [float(i)],
                'mask': [1],
                'next_state': [np.array([i + 1], dtype=np.float32)]
            }
            self.wrapper.feed(data)
            time.sleep(0.01)  # Allow a brief moment for the asynchronous process.

    def tearDown(self):
        self.wrapper.close()

    def test_wrapper_sample(self):
        transition = self.wrapper.sample()

        # Check shapes of sampled arrays.
        self.assertEqual(transition.state.shape[0], self.batch_size)
        self.assertEqual(transition.action.shape[0], self.batch_size)
        
        print(transition.reward)
        # Check that reward values are within the expected range (from 5 to 14).
        for r in transition.reward:            
            self.assertTrue(5 <= r.item() <= 14)

class TestReplayProxy(unittest.TestCase):
    def setUp(self):
        self.memory_size = 10
        self.batch_size = 4
        self.keys = ['state', 'action', 'reward', 'mask', 'next_state']
        self.replay_kwargs = {
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "n_step": 1,
            "discount": 0.99,
            "history_length": 1,
            "keys": self.keys
        }

        # Create an asynchronous ReplayWrapper.
        self.wrapper = ReplayWrapper(UniformReplay, self.replay_kwargs, asynchronous=True)
        
        # Wait for a "ready" signal from the worker.
        ready, _ = self.wrapper.pipe.recv()
        self.assertEqual(ready, "ready")
        
        # Create a ReplayProxy using the pipe from the wrapper.
        self.proxy = ReplayProxy(self.wrapper.pipe)
        
        # Feed 15 dummy transitions via the proxy.
        for i in range(15):
            data = {
                'state': [np.array([i], dtype=np.float32)],
                'action': [np.array([i + 0.1], dtype=np.float32)],
                'reward': [float(i)],
                'mask': [1],
                'next_state': [np.array([i + 1], dtype=np.float32)]
            }
            self.proxy.feed(data)
            time.sleep(0.01)

    def tearDown(self):
        self.wrapper.close()

    def test_proxy_sample(self):
        transition = self.proxy.sample()
        
        # Verify that each field has the expected batch dimension.
        self.assertEqual(transition.state.shape[0], self.batch_size)
        self.assertEqual(transition.action.shape[0], self.batch_size)
        
        # Check that rewards are within the expected range.
        for r in transition.reward:
            self.assertTrue(5 <= r.item() <= 14)

if __name__ == '__main__':
    # Set the multiprocessing start method.
    mp.set_start_method('spawn', force=True)
    unittest.main()
