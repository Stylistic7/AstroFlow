# TODO: Maybe remove this low-level interface?
# TODO: Exposing FileList is a wart on the API.
from .classes import ShellScriptTask, FileList, ViewIdList
from .shell_script import _clear_cache
from .abstract_image import AbstractImage

from .decorators import PixInsight
from .context import append_task, get_next_process_number, get_tmp_dir
from .open_images import OpenImages
from .save_images import SaveImages
from .close_images import CloseImages
from .close_all_windows import CloseAllWindows
from .image_solver import ImageSolver
from .automatic_background_extractor import AutomaticBackgroundExtractor
from .wbpp import WeightedBatchPreprocessing
from .mosaic_by_coordinates import MosaicByCoordinates
from .blurxterminator import BlurXTerminator
from .integer_resample import IntegerResample
from .fast_rotation import FastRotation
from .photometric_mosaic import PhotometricMosaic