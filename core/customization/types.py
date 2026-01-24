from stacks.composite_stack import CompositeStack
from hashing import hash_list_int


class CustomizedStack(CompositeStack):
    pass


class CustomizedStacksDict(dict[str, CustomizedStack]):
    def __hash__(self):
        return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

    def __str__(self):
        return f"CustomizedStacksDict({len(self)})"
