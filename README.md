# Tricky generators

Some papers introduced molecule generators that create molecules based on the predictions of a learned property prediction model.

Here we show that this might go wrong.

# TODOs:
- Add figures jpg

Think about switching to models with higher predictive performance

# New names
- Split 1, model 1 -> Optimization score (OS)
- Split 1, model 2 -> Model control score (MCS)
- Split 2 -> Data control score (DCS)

MCS is conflicting with maximum common substructure though -> CSM

# Refactoring
- Changed results -> results/goal_directed
- distribution_results -> results/distribution
- trainset_predictions -> results/predictions


Steps to reproduce the paper:
- run_goal_directed.py
- calculate trainset predictions
- summary plots
- nearest neighbours# mgenerators-failure-modes
