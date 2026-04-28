"""Latent RL: Plataforma experimental para RL financiero con representaciones latentes."""

__version__ = "0.1.0"

from latent_rl.data import CSVDataLoader, DataPreprocessor, YahooFinanceLoader
from latent_rl.envs import FinancialEnv
from latent_rl.representations import MLPLatentEncoder
from latent_rl.evaluation import FinancialMetrics
from latent_rl.agents import BaseAgent, RandomAgent, BuyAndHoldAgent, DQNAgent, ReplayBuffer, LatentDQNAgent
from latent_rl.pretraining import AutoencoderTrainer

__all__ = [
    "CSVDataLoader",
    "DataPreprocessor",
    "YahooFinanceLoader",
    "FinancialEnv",
    "MLPLatentEncoder",
    "FinancialMetrics",
    "BaseAgent",
    "RandomAgent",
    "BuyAndHoldAgent",
    "DQNAgent",
    "ReplayBuffer",
    "LatentDQNAgent",
    "AutoencoderTrainer",
]