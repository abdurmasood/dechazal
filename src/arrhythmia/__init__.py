"""Inter-patient ECG beat classification on MIT-BIH with AAMI classes."""

import os

# Keras reads its backend at first import; default to JAX unless the caller chose one.
os.environ.setdefault("KERAS_BACKEND", "jax")
