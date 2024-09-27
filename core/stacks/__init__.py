from typing import NewType

from .simple_stack import SimpleStack
from .composite_stack import CompositeStack, SimpleStacksDict
from .post_process import post_process_stack_artifacts
from .artifacts import output_filename, stack_files_artifacts
from .state import update_state
from .validation import validate_stacks
from .astrometric_attributes import AstrometricAttributes
from .imaging_attributes import ImagingAttributes
from .state_attributes import AstroFlowStateAttributes
from .types import StackHash, FrameHash

CompositeStacksDict = dict[str, CompositeStack]
