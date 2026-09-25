"""The filter itself: one implementation, and one optional backend.

spamlib.py is the pipeline every entry point runs -- cleaning, features,
training, prediction. embeddings.py is the transformer backend it will use
if asked and the dependency is installed, and nothing imports it directly.
"""
