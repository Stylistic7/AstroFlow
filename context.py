from dataclasses import dataclass, field, asdict
import datetime


@dataclass
class Unionable:

  def __or__(self, other):
    return self.__class__(**asdict(self) | other)


@dataclass
class Context(Unionable):
  input_dirs: list[str] = field(default_factory=list)
  output_dir: str = None

  default_filter: str = None

  image_hashes: dict = None
  fits_headers: dict = None
  plate_solutions: dict = None
  quarantined_files: list = None

  regroup: bool = False

  spatial_groups: list = None

  stacks: dict = None
  targets: dict = None
  mosaics: dict = None

  cosine_similarity_threshold: float = None
  min_component_members: int = None

  max_run_gap: datetime.timedelta = None
  min_frames_in_run: int = None
  min_frames_in_stack: int = None

  coordinates = None

  run_everything: bool = False
  shopping_cart: list = field(default_factory=list)
