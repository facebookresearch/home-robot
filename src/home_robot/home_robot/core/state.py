import sophus as sp
from dataclasses import dataclass


@dataclass
class ManipulatorBaseParams:
    se3_base: sp.SE3
