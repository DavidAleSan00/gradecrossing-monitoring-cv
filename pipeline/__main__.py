"""Allow `python -m pipeline <command>`."""

import sys

from .cli import main

sys.exit(main())
