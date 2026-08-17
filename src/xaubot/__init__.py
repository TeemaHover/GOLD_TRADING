"""xaubot -- machine-learning trading system for XAUUSD.

Layering (strictly downward dependencies):

``config`` / ``core`` -> ``data`` -> ``features`` -> ``labels`` -> ``datasets``
-> ``models`` -> ``training`` -> ``inference`` -> ``signals`` -> ``risk``
-> ``execution`` -> ``evaluation`` -> ``service``/``dashboard``

See docs/ARCHITECTURE.md for the full design.
"""

from __future__ import annotations

__version__ = "0.1.0"
