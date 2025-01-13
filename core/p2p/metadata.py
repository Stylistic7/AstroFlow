import h5py
import io
import numpy as np
import json
from typing import Any
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from datetime import datetime
from stacks.composite_stack import CompositeStack
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore


def _ndarray_to_json(array: np.ndarray) -> dict:
  return {
      "data": array.tolist(),  # Convert data to a list
      "shape": array.shape,  # Store the shape
      "dtype": str(array.dtype),  # Store the data type
  }


def _json_to_ndarray(serialized: dict) -> np.ndarray:
  return np.array(serialized["data"],
                  dtype=serialized["dtype"]).reshape(serialized["shape"])


def _json_to_single_spherical_polygon(data: dict) -> SingleSphericalPolygon:
  points = _json_to_ndarray(data['points'])
  inside = _json_to_ndarray(data['inside'])
  return SingleSphericalPolygon(points, inside)


def _single_spherical_polygon_to_json(polygon: SingleSphericalPolygon) -> dict:
  return json.dumps({
      'points': _ndarray_to_json(polygon.points),
      'inside': _ndarray_to_json(polygon.inside)
  })


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
  # Example image data (randomized for demo purposes)
  image_data = {
      "image1": np.random.rand(256, 256),  # Grayscale image
      "image2": np.random.rand(256, 256, 3),  # RGB image
  }

  # Create the HDF5 file
  with h5py.File(path, "w") as h5file:
    # Store stack data
    stack_group = h5file.create_group("stacks")
    for key, stack in stacks.items():
      # Serialize the Stack object to JSON
      stack_json = json.dumps(asdict(stack), cls=CustomJSONEncoder)
      stack_group.create_dataset(key, data=stack_json)

    # Store image data
    image_group = h5file.create_group("images")
    for key, image in image_data.items():
      image_group.create_dataset(key, data=image, compression="gzip")


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
              (f.type for f in fields(CompositeStack) if f.name == field_name),
              None)

          if field_type == SingleSphericalPolygon:
            stack_dict[field_name] = _json_to_single_spherical_polygon(
                field_value)
          elif field_type == datetime:
            stack_dict[field_name] = _json_to_datetime(field_value)
          elif isinstance(
              field_value,
              str) and field_value.startswith("["):  # Handle arrays as strings
            stack_dict[field_name] = _json_to_ndarray(field_value)

        # Recreate the Stack object
        stacks[key] = CompositeStack(**stack_dict)

      # Read image data from the "images" group
      image_group = h5file["images"]
      for key, image_data in image_group.items():
        # Store images as NumPy arrays
        stacks[key] = np.array(image_data)

    return stacks
