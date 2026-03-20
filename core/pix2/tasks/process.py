"""Factory for simple PixInsight process tasks.

Most PixInsight processes follow the same pattern: create a process instance,
set parameters, and execute on one or more views. This module provides a
factory that generates task functions for these simple cases.
"""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step
from pix2.tasks.js import execute_on_views


def make_process_task(
    process_name: str,
    default_parameters: dict[str, Any],
    tag: str | None = None,
):
    """Create a task function for a simple execute-on-views process.

    The returned function takes an Image (or list of Images) and optional
    kwargs to override default parameters, appends a step to the recipe,
    and returns new Image(s) sharing the same recipe.
    """

    def task(image: Image, **kwargs: Any) -> Image:
        js = execute_on_views(
            process_name,
            default_parameters | kwargs,
            [image.view_id],
        )
        image.recipe.add_step(Step(javascript=js, name=process_name))
        return Image(
            view_id=image.view_id,
            recipe=image.recipe,
            filter=image.filter,
        )

    task.__name__ = process_name
    task.__qualname__ = process_name
    return task
