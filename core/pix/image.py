"""Lazy image type for pix.

An Image represents the result of a PixInsight operation (or chain of
operations) that hasn't been executed yet. It looks and feels like a
normal object, but execution is deferred until save_to() is called.
"""
from __future__ import annotations

from pathlib import Path

from pix.recipe import Recipe


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
    def open(cls, filename: Path | str) -> "Image":
        """Create a lazy image from a file on disk.

        This does not read the file — it just records that this file
        will be opened when the recipe is executed.
        """
        from pix.tasks.open_images import read_filter_from_header

        filename = Path(filename)
        recipe = Recipe(input_filenames=[filename])
        filter = read_filter_from_header(filename)
        view_id = f"V_{filename.stem}"
        return cls(view_id=view_id, recipe=recipe, filter=filter)

    @classmethod
    def open_many(cls, filenames: list[Path | str]) -> "ImageList":
        """Create lazy images from multiple files on disk."""
        filenames = [Path(f) for f in filenames]
        recipe = Recipe(input_filenames=filenames)
        images = []
        for filename in filenames:
            from pix.tasks.open_images import read_filter_from_header
            filter = read_filter_from_header(filename)
            view_id = f"V_{filename.stem}"
            images.append(cls(view_id=view_id, recipe=recipe, filter=filter))
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
