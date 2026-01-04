# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ---------- presets (AAL2 cortical names) ----------
_VISUAL_CORE = (
    "Calcarine", "Cuneus", "Lingual"
)

_VISUAL_EXTENDED = (
    "Occipital_Sup", "Occipital_Mid", "Occipital_Inf", "Fusiform"
)

_MOTOR_CORE = (
    "Precentral",
)

_MOTOR_EXTENDED = (
    "Supp_Motor_Area", "Paracentral_Lobule"
)

# Sometimes people treat these as "motor-related" but they are not motor output:
_SOMATOSENSORY = (
    "Postcentral",
)

# region_names should be your LRLR-ordered AAL2 cortical list (length 80).
region_names = tuple([
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
    "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
    "Rectus_L", "Rectus_R",
    "Insula_L", "Insula_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R",
    "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R",
    "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R",
    "Fusiform_L", "Fusiform_R",
    "Temporal_Sup_L", "Temporal_Sup_R",
    "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    "Heschl_L", "Heschl_R",
    "Temporal_Sup_Oper_L", "Temporal_Sup_Oper_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    "Occipital_Medial_L", "Occipital_Medial_R",
])


# Common AAL2 naming uses suffixes: _L / _R
def _canon_lr(name: str, hemi: str) -> str:
    if hemi not in ("L", "R"):
        raise ValueError(f"hemi must be 'L' or 'R', got {hemi!r}")
    return f"{name}_{hemi}"


@dataclass(frozen=True)
class IOIndexConfig:
    # Visual selection
    visual: str = "core"  # "core" | "extended" | "core+extended"
    # Motor selection
    motor: str = "core"  # "core" | "extended" | "core+extended"
    # Hemisphere selection
    hemisphere: str = "both"  # "both" | "L" | "R"
    # Whether to include somatosensory cortex in outputs (usually False)
    include_somatosensory: bool = False
    # Strict checking: raise if any target region not found in region_names
    strict: bool = True


def get_io_region_indices(
    config: IOIndexConfig = IOIndexConfig(),
) -> Dict[str, List[int]]:
    """
    Return input/output region indices for a whole-brain model based on AAL2 cortical names.

    Parameters
    ----------
    config:
        IOIndexConfig controlling which populations are selected.

    Returns
    -------
    dict with keys:
        - "input_idx": indices for visual input regions
        - "output_idx": indices for motor output regions
        - "input_names": selected visual region names (resolved to L/R)
        - "output_names": selected motor region names (resolved to L/R)

    Notes
    -----
    - Visual "core": Calcarine, Cuneus, Lingual
    - Visual "extended": Occipital_Sup/Mid/Inf, Fusiform
    - Motor "core": Precentral
    - Motor "extended": Supp_Motor_Area, Paracentral_Lobule
    """
    # Build name -> index map
    name_to_idx = {n: i for i, n in enumerate(region_names)}

    def pick_hemi(basenames: Iterable[str]) -> List[str]:
        if config.hemisphere == "both":
            hemis = ("L", "R")
        elif config.hemisphere in ("L", "R"):
            hemis = (config.hemisphere,)
        else:
            raise ValueError(f"hemisphere must be 'both'|'L'|'R', got {config.hemisphere!r}")
        out: List[str] = []
        for b in basenames:
            for h in hemis:
                out.append(_canon_lr(b, h))
        return out

    def resolve_visual_basenames() -> Tuple[str, ...]:
        if config.visual == "core":
            return _VISUAL_CORE
        if config.visual == "extended":
            return _VISUAL_EXTENDED
        if config.visual in ("core+extended", "extended+core", "all"):
            return _VISUAL_CORE + _VISUAL_EXTENDED
        raise ValueError(f"visual must be 'core'|'extended'|'core+extended', got {config.visual!r}")

    def resolve_motor_basenames() -> Tuple[str, ...]:
        motor = ()
        if config.motor == "core":
            motor = _MOTOR_CORE
        elif config.motor == "extended":
            motor = _MOTOR_EXTENDED
        elif config.motor in ("core+extended", "extended+core", "all"):
            motor = _MOTOR_CORE + _MOTOR_EXTENDED
        else:
            raise ValueError(f"motor must be 'core'|'extended'|'core+extended', got {config.motor!r}")

        if config.include_somatosensory:
            motor = motor + _SOMATOSENSORY
        return motor

    input_names = pick_hemi(resolve_visual_basenames())
    output_names = pick_hemi(resolve_motor_basenames())

    missing_in = [n for n in input_names if n not in name_to_idx]
    missing_out = [n for n in output_names if n not in name_to_idx]
    if config.strict and (missing_in or missing_out):
        raise KeyError(
            "Some requested regions were not found in region_names.\n"
            f"Missing input regions: {missing_in}\n"
            f"Missing output regions: {missing_out}\n"
            "Tip: check naming (AAL2 uses e.g. 'Occipital_Mid_L') and ensure you're using cortical-only list."
        )

    input_idx = [name_to_idx[n] for n in input_names if n in name_to_idx]
    output_idx = [name_to_idx[n] for n in output_names if n in name_to_idx]

    return {
        "input_idx": np.asarray(input_idx, dtype=np.int32),
        "output_idx": np.asarray(output_idx, dtype=np.int32),
        "input_names": [region_names[i] for i in input_idx],
        "output_names": [region_names[i] for i in output_idx],
    }


if __name__ == "__main__":
    # Default: visual core, motor core, both hemispheres
    io0 = get_io_region_indices()
    print("Default:")
    print('  - input names:', io0["input_names"])
    print('  - output names:', io0["output_names"])

    # Extended visual + extended motor planning, both hemispheres
    cfg = IOIndexConfig(visual="core+extended", motor="core+extended", hemisphere="both")
    io1 = get_io_region_indices(cfg)
    print("Extended:")
    print('  - input names:', io1["input_names"])
    print('  - output names:', io1["output_names"])

    # Left hemisphere only
    cfg_L = IOIndexConfig(visual="core", motor="core", hemisphere="L")
    io2 = get_io_region_indices(cfg_L)
    print("Left-only:")
    print('  - input names:', io2["input_names"])
    print('  - output names:', io2["output_names"])
