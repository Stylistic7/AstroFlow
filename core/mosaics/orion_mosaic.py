import datetime

from context import context
from testing.test_frames import copy_testdata_frames, make_synthetic_frames


def add_orion_mosaic_images(panes=[1, 2, 3, 4], add_calibration_frames=True, **kwargs):
    if 1 in panes:
        copy_testdata_frames(
            target_name="Orion Pane 1",
            start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
    if 2 in panes:
        copy_testdata_frames(
            target_name="Orion Pane 2",
            start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
    if 3 in panes:
        copy_testdata_frames(
            target_name="Orion Pane 3",
            start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
    if 4 in panes:
        copy_testdata_frames(
            target_name="Orion Pane 4",
            start_datetime=datetime.datetime(2024, 2, 5, 19, 29, 43),
            dest_dir=context.input_dirs[0],
            **kwargs,
        )

    if add_calibration_frames:
        make_synthetic_frames(
            prefix="dark",
            number=5,
            image_type="Dark",
            start_datetime=datetime.datetime(2024, 2, 4, 18, 29, 43),
            fill_value=100,
            instrument="ZWO ASI6200MM Pro",
            width_pixels=9576,
            height_pixels=6388,
            temperature=-20,
            exposure=300,
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
        make_synthetic_frames(
            prefix="bias",
            number=7,
            image_type="Bias",
            start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
            fill_value=50,
            instrument="ZWO ASI6200MM Pro",
            width_pixels=9576,
            height_pixels=6388,
            temperature=-20,
            exposure=300,
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
        make_synthetic_frames(
            prefix="flat",
            number=11,
            image_type="Flat",
            start_datetime=datetime.datetime(2024, 2, 6, 18, 29, 43),
            fill_value=50,
            instrument="ZWO ASI6200MM Pro",
            width_pixels=9576,
            height_pixels=6388,
            temperature=-20,
            exposure=300,
            # The Orion lights are FOCALLEN=350; a flat from another optical
            # train is refused outright (`_chronological_flat`), so the
            # fixture's flat must be shot through the same one.
            focal_length_m=0.35,
            dest_dir=context.input_dirs[0],
            **kwargs,
        )
