"""
Pytest configuration for test environment.
Configures matplotlib to use non-interactive 'Agg' backend to avoid Tkinter GUI issues in headless / test environments.
"""

import matplotlib
matplotlib.use("Agg")
