"""Deferring a walk, so its cost has a ceiling.

    optimize.py   the context that turns deferral on, and the budget it sets
    plan.py       a walk described but not taken, and how it is run
    planner.py    what a plan may do with a condition, read off the condition
    stream.py     reductions that take a planned walk a slice at a time
"""
