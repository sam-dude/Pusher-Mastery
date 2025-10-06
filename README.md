# Pusher Mastery - Notebook-Based Structure
This is a deep dive into robotic manipulation using Reinforcement Learning. In this experiment I learn to try different deep learning algorithm for Pusher environment provided by Gymnasium

## Getting Started
1. Activate environment: \`source pusher_env/bin/activate\`
2. Launch Jupyter: \`jupyter lab\`
3. Open \`notebooks/00_setup_and_test.ipynb\`


## 📁 Project Layout

```
pusher_mastery/
├── README.md
├── requirements.txt
├── setup.py
│
├── notebooks/                          # Main learning happens here
│   ├── 00_setup_and_test.ipynb       # Environment setup
│   ├── 01_environment_exploration.ipynb
│   ├── 02_heuristic_policy.ipynb
│   ├── 03_sac_implementation.ipynb
│   ├── 04_sac_training.ipynb
│   ├── 05_reward_engineering.ipynb
│   ├── 06_curriculum_learning.ipynb
│   ├── 07_algorithm_comparison.ipynb
│   └── 08_analysis_and_visualization.ipynb
│
├── src/                                # Clean implementations
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sac.py                     # Extracted from notebook
│   │   ├── td3.py
│   │   └── ppo.py
│   ├── environments/
│   │   ├── __init__.py
│   │   └── wrappers.py                # Custom env wrappers
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       ├── logging.py
│       └── replay_buffer.py
│
├── scripts/                            # Standalone scripts
│   ├── train_sac.py                   # For long training runs
│   ├── train_td3.py
│   ├── evaluate.py
│   └── create_videos.py
│
├── results/                            # Saved results
│   ├── figures/
│   ├── videos/
│   ├── logs/
│   └── tables/
│
├── checkpoints/                        # Model checkpoints
│   ├── sac/
│   ├── td3/
│   └── ppo/
│
├── docs/                               # Documentation
│   ├── lessons_learned.md
│   ├── troubleshooting.md
│   └── blog_drafts/
│
└── data/                               # Saved datasets
    └── demonstrations/
```


## Notebooks Overview

1. **00_setup_and_test.ipynb** - Verify installation
2. **01_environment_exploration.ipynb** - Understand Pusher
3. **02_heuristic_policy.ipynb** - Build baseline
4. **03_sac_implementation.ipynb** - Implement SAC
5. **04_sac_training.ipynb** - Train agent
6. **05_reward_engineering.ipynb** - Optimize rewards
7. **06_curriculum_learning.ipynb** - Speed up learning
8. **07_algorithm_comparison.ipynb** - Compare algorithms
9. **08_analysis_and_visualization.ipynb** - Deep analysis

## Author

Samuel Ibiyemi

## License

MIT
