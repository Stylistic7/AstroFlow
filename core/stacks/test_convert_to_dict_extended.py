"""Extended tests for convert_to_dict."""
from dataclasses import dataclass

from stacks.convert_to_dict import convert_to_dict


@dataclass
class Inner:
    x: int
    y: str


@dataclass
class Outer:
    name: str
    inner: Inner
    items: list


def test_simple_dataclass():
    result = convert_to_dict(Inner(x=1, y="hello"))
    assert result == {"x": 1, "y": "hello"}


def test_nested_dataclass():
    obj = Outer(name="test", inner=Inner(x=1, y="a"), items=[1, 2, 3])
    result = convert_to_dict(obj)
    assert result == {"name": "test", "inner": {"x": 1, "y": "a"}, "items": [1, 2, 3]}


def test_omit_paths():
    obj = Outer(name="test", inner=Inner(x=1, y="a"), items=[1, 2])
    result = convert_to_dict(obj, omit_paths=[["inner"]])
    assert "inner" not in result
    assert result["name"] == "test"


def test_omit_nested_path():
    obj = Outer(name="test", inner=Inner(x=1, y="a"), items=[1, 2])
    result = convert_to_dict(obj, omit_paths=[["inner", "y"]])
    assert result["inner"] == {"x": 1}


def test_transform():
    obj = Inner(x=42, y="hello")
    result = convert_to_dict(obj, transform=lambda v: v * 2 if isinstance(v, int) else v)
    assert result == {"x": 84, "y": "hello"}


def test_plain_dict():
    result = convert_to_dict({"a": 1, "b": {"c": 2}})
    assert result == {"a": 1, "b": {"c": 2}}


def test_list_of_dataclasses():
    items = [Inner(x=1, y="a"), Inner(x=2, y="b")]
    result = convert_to_dict(items)
    assert result == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
