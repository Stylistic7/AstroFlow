from stacks.composite_stack import CompositeStack
from hashing import hash_list_int

import units


class CustomizedStack(CompositeStack):
    def __str__(self) -> str:
        return (
            f"CustomizedStack(Stable hash {self.stable_hash}, "
            f"Content hash {self.content_hash}, "
            f"Target {self.name}, "
            f"RA {units.ra_to_hms(self.center[0])}, "
            f"Dec {units.dec_to_dms(self.center[1])}, "
            f'Scale {self.image_scale:.2f}"/px, '
            f"PA {self.rotation_angle:.2f} deg, "
            f"Camera {self.camera}, "
            f"Exposure {self.exposure_s} s, "
            f"Integration {self.integration_s} s, "
            f"Filter {self.filter}, "
            f"Temp {self.mean_temperature:.02f} degC)"
        )


class CustomizedStacksDict(dict[str, CustomizedStack]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"CustomizedStacksDict({len(self)})"
