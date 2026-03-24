import math
from stacks.composite_stack import CompositeStack
from stacks.types import StackHash
from hashing import hash_list_int
from cache.context_ref import ContextRef

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
            f"PA {math.degrees(self.rotation_angle):.2f} deg, "
            f"Camera {self.camera}, "
            f"Exposure {self.exposure} s, "
            f"Integration {self.integration} s, "
            f"Filter {self.filter}, "
            f"Temp {self.mean_temperature:.02f} degC)"
        )


class CustomizedStacksDict(dict[StackHash, ContextRef["CustomizedStack"]]):
    def __hash__(self):
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"CustomizedStacksDict({len(self)})"
