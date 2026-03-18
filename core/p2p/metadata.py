import h5py
import io
import numpy as np
import json
from typing import Any
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from datetime import datetime
from stacks.composite_stack import CompositeStack
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore

from context import context


def _ndarray_to_json(array: np.ndarray) -> dict:
    return {
        "data": array.tolist(),  # Convert data to a list
        "shape": array.shape,  # Store the shape
        "dtype": str(array.dtype),  # Store the data type
    }


def _json_to_ndarray(serialized: dict) -> np.ndarray:
    return np.array(serialized["data"], dtype=serialized["dtype"]).reshape(
        serialized["shape"]
    )


def _json_to_single_spherical_polygon(data: dict) -> SingleSphericalPolygon:
    points = _json_to_ndarray(data["points"])
    inside = _json_to_ndarray(data["inside"])
    return SingleSphericalPolygon(points, inside)


def _single_spherical_polygon_to_json(polygon: SingleSphericalPolygon) -> dict:
    return json.dumps(
        {
            "points": _ndarray_to_json(polygon.points),
            "inside": _ndarray_to_json(polygon.inside),
        }
    )


def _json_to_datetime(date_string: str) -> datetime:
    return datetime.fromisoformat(date_string)


# Custom JSON encoder with type tagging
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, SingleSphericalPolygon):
            return _single_spherical_polygon_to_json(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def stacks_to_hdf5(stacks, path):
    with h5py.File(path, "w") as h5file:
        stack_group = h5file.create_group("stacks")
        for key, stack in stacks.items():
            stack_json = json.dumps(asdict(stack), cls=CustomJSONEncoder)
            stack_group.create_dataset(key, data=stack_json)

        subframe_group = h5file.create_group("stack_subframe_thumbnail_images")
        master_group = h5file.create_group("stack_master_thumbnail_images")
        for stable_hash in stacks:
            key = ("stack_subframe_thumbnail_image", stable_hash)
            if context.contains(key):
                image = context[key]
                print(f"### create_dataset key {key} type(image) {type(image)}")
                subframe_group.create_dataset(
                    stable_hash, data=context[key], compression="gzip"
                )
            key = ("stack_master_thumbnail_image", stable_hash)
            if context.contains(key):
                image = context[key]
                print(f"### create_dataset key {key} type(image) {type(image)}")
                master_group.create_dataset(
                    stable_hash, data=context[key], compression="gzip"
                )


def hdf5_to_stacks(data: str) -> dict[str, Any]:
    stacks = {}

    with io.BytesIO() as buf:
        buf.write(data)
        buf.seek(0)

        with h5py.File(buf, "r") as h5file:
            # Read stack data from the "stacks" group
            stack_group = h5file["stacks"]
            for key, stack_json in stack_group.items():
                # Deserialize the JSON back into a dictionary
                stack_dict = json.loads(stack_json[()])

                # Deserialize each field based on its type
                for field_name, field_value in stack_dict.items():
                    # Handle custom types based on field name or type
                    field_type = next(
                        (
                            f.type
                            for f in fields(CompositeStack)
                            if f.name == field_name
                        ),
                        None,
                    )

                    if field_type == SingleSphericalPolygon:
                        stack_dict[field_name] = _json_to_single_spherical_polygon(
                            field_value
                        )
                    elif field_type == datetime:
                        stack_dict[field_name] = _json_to_datetime(field_value)
                    elif isinstance(field_value, str) and field_value.startswith(
                        "["
                    ):  # Handle arrays as strings
                        stack_dict[field_name] = _json_to_ndarray(field_value)

                # Recreate the Stack object
                stacks[key] = CompositeStack(**stack_dict)

            subframe_group = h5file["stack_subframe_thumbnail_images"]
            for key, image_data in subframe_group.items():
                context[("stack_subframe_thumbnail_image", key)] = np.array(image_data)

            master_group = h5file["stack_master_thumbnail_images"]
            for key, image_data in master_group.items():
                context[("stack_master_thumbnail_image", key)] = np.array(image_data)

        from cache.context_ref import ContextRef
        from stacks.composite_stack import CompositeStacksDict
        stacks_refs = CompositeStacksDict({
            k: ContextRef.store(("stacks", k), v) for k, v in stacks.items()
        })
        context.stacks = stacks_refs
        return stacks
