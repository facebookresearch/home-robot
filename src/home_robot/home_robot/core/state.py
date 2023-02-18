from dataclasses import dataclass

import sophus as sp


@dataclass
class ManipulatorBaseParams:
    se3_base: sp.SE3
