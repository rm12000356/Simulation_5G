"""Compatibility wrapper for modulation functions.

This file re-exports the main functions from `M_QAM_modulation.py` so
other scripts can import `from modulation import ...` as they used to.
"""

try:
	# Normal import when running scripts from the repository root
	from M_QAM_modulation import (
		modulation_MQAM,
		plot_constellation,
		demodulate_MQAM,
		generate_all_points_in_constellation,
	)
except Exception:
	# Fallback to relative import if this package is used as a module
	from .M_QAM_modulation import (
		modulation_MQAM,
		plot_constellation,
		demodulate_MQAM,
		generate_all_points_in_constellation,
	)

__all__ = [
	"modulation_MQAM",
	"plot_constellation",
	"demodulate_MQAM",
	"generate_all_points_in_constellation",
]
