"""Pattern detector implementations."""

from src.detectors.d01 import D01
from src.detectors.d02 import D02
from src.detectors.d03 import D03
from src.detectors.d04 import D04
from src.detectors.d05 import D05
from src.detectors.d06 import D06
from src.detectors.d07 import D07
from src.detectors.d08 import D08

DETECTORS = (D01, D02, D03, D04, D05, D06, D07, D08)

__all__ = ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "DETECTORS"]
