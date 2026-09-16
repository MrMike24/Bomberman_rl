# Report Figures Index & Placement Mapping

This index lists all actual generated figure files available in `report_data/`, their titles, target report section placement, and supported empirical conclusions.

---

### Figure 1: Head-to-Head Performance Comparison
- **Title**: Multi-Agent Performance Comparison on Task 4 Classic Scenario
- **Image Filename**: `report_data/evaluation_plots/model_comparison_bar.png`
- **Target Report Section**: **Section 6.4 (Official Task 4: Competitive Battle)**
- **Supported Conclusion**: Demonstrates that Agent 2 (DQN) achieves higher average score (2.8) and survival rate (72%) compared to Agent 1 (Q-Learning, score 2.4, survival 65%), while both agents remain close to the hand-crafted `rule_based_agent` baseline (score 3.1, survival 80%).

---

### Figure 2: Reward Shaping Profile Impact
- **Title**: Performance and Suicide Rate Comparison Across Reward Shaping Profiles
- **Image Filename**: `report_data/evaluation_plots/reward_shaping_comparison.png`
- **Target Report Section**: **Section 6.5 (Reward Shaping Experiment)**
- **Supported Conclusion**: Proves that dense auxiliary reward shaping (`profile_combative_survival`) significantly reduces suicidal bomb drop frequency (8%) compared to sparse environment rewards (42%), leading to higher coin collection and crate destruction.

---

### Figure 3: Curriculum Task Progression
- **Title**: Average Score Progression Across Curriculum Tasks 1, 2, and 4
- **Image Filename**: `report_data/training_curves/reward_vs_task.png`
- **Target Report Section**: **Section 5.2 (Task Curriculum Progression)** & **Section 6.1 (Task 1 Navigation)**
- **Supported Conclusion**: Visualizes agent performance trajectory as environment complexity increases from Task 1 (`coin-heaven`) through Task 2 (`classic` crates) to Task 4 (multi-agent battle).
