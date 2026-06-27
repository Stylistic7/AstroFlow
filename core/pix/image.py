"""Lazy image type for pix.

An Image represents the result of a PixInsight operation (or chain of
operations) that hasn't been executed yet. It looks and feels like a
normal object, but execution is deferred until save_to() is called.
"""
from __future__ import annotations

import xxhash
from pathlib import Path

from pix.recipe import Recipe


def view_id_for(filename: Path) -> str:
    """Deterministic, short, PixInsight-safe view ID from a filename."""
    return f"V_{xxhash.xxh64(str(filename)).hexdigest()[:12]}"


class Image:
    """A lazy reference to an image that will exist after PixInsight runs.

    User code creates Images via Image.open() and passes them through
    task functions like BlurXTerminator(). These calls build up a Recipe
    internally. Nothing is executed until save_to() is called.
    """

    def __init__(self, view_id: str, recipe: Recipe, filter: str | None = None):
        self.view_id = view_id
        self.recipe = recipe
        self.filter = filter
        recipe.register_output(self)

    @classmethod
    def open(cls, filename: Path | str, filter: str | None = None) -> "Image":
        """Create a lazy image from a file on disk.

        This does not read the file — it just records that this file
        will be opened when the recipe is executed. We do, however,
        check that the file exists at construction time: a missing
        input here will eventually launch PixInsight and hang on
        ImageWindow.open until the script timeout, which is wasteful
        and obscures the real cause. Failing fast lets Ray's retry
        machinery and the next graph run handle it cleanly.

        ``filter`` is an opaque label the caller attaches (e.g. the
        FITS FILTER keyword, or the grouping value from upstream
        metadata). pix never introspects the file for it — that's
        the caller's concern.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"pix.Image.open: input does not exist: {filename}")
        recipe = Recipe(input_filenames=[filename])
        view_id = view_id_for(filename)
        return cls(view_id=view_id, recipe=recipe, filter=filter)

    @classmethod
    def open_many(
        cls,
        filenames: list[Path | str],
        filters: list[str | None] | None = None,
    ) -> "ImageList":
        """Create lazy images from multiple files on disk.

        Same fail-fast behavior as `open`: any missing input raises
        a FileNotFoundError listing every missing path, before any
        PixInsight script work begins.

        ``filters``, if supplied, must be a parallel list of filter
        labels (one per filename); otherwise every image's filter is
        ``None``. pix never introspects the files for filter values.
        """
        filenames = [Path(f) for f in filenames]
        missing = [str(f) for f in filenames if not f.exists()]
        if missing:
            raise FileNotFoundError(f"pix.Image.open_many: inputs do not exist: {missing}")
        if filters is None:
            filters = [None] * len(filenames)
        elif len(filters) != len(filenames):
            raise ValueError(
                f"pix.Image.open_many: filters has length {len(filters)} "
                f"but filenames has length {len(filenames)}"
            )
        recipe = Recipe(input_filenames=filenames)
        images = [
            cls(view_id=view_id_for(f), recipe=recipe, filter=flt)
            for f, flt in zip(filenames, filters)
        ]
        return ImageList(images)

    def save_to(self, path: Path | str, mock: bool = False) -> None:
        """Materialize this image and save it to the given path.

        If the recipe hasn't been executed yet, this triggers execution
        (running PixInsight or mocking). If it has already been executed
        (e.g. by a sibling image's save_to), this is just a file copy.
        """
        path = Path(path)
        self.recipe.materialize(mock=mock)
        self.recipe.copy_output(self, path)


class ImageList(list):
    """A list of Images, with filter-based lookup."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        if isinstance(key, slice):
            # `images[1:]` (e.g. StarAlignment's reference/targets split) must
            # return a sub-list, not fall through to filter-by-string lookup
            # (which would call `.lower()` on the slice object).
            return ImageList(super().__getitem__(key))
        # String key: find by filter
        from pix.tasks.filters import normalize_filter
        for image in self:
            if normalize_filter(image.filter) == normalize_filter(key):
                return image
        return None

    def __contains__(self, filters):
        if isinstance(filters, str):
            from pix.tasks.filters import normalize_filter
            return all(
                any(normalize_filter(img.filter) == normalize_filter(f) for img in self)
                for f in filters
            )
        return super().__contains__(filters)

    def channels(self, filters: str) -> list[Image]:
        return [self[f] for f in filters]
