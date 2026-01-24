from stacks.simple_stack import SimpleStack


def get_content_stacks_of_type(stacks, image_type):
    return [
        s
        for s in stacks.values()
        if s.image_type == image_type and isinstance(s, SimpleStack)
    ]
