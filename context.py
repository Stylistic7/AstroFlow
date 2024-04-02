from dataclasses import dataclass, field


@dataclass
class Context:
  input_dirs: list[str] = field(default_factory=list)
  output_dir: str = None

  default_filter: str = None

  image_hashes: dict = None
  fits_headers: dict = None
  plate_solutions: dict = None
  quarantined_files: list = None

  needs_regrouping: bool = False

  spatial_groups: list = None

  stacks: dict = None
  targets: dict = None
  mosaics: dict = None

  cosine_similarity_threshold = None
  min_component_members = None

  coordinates = None
