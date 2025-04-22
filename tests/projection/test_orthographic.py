import numpy as np
import pytest
from numpy.typing import NDArray
from flatprot.projection.orthographic import (
    OrthographicProjection,
    OrthographicProjectionParameters,
)

# Define expected floating point type for numpy arrays in tests
NpFloatArray = NDArray[np.float64]


@pytest.fixture
def simple_cube() -> NpFloatArray:
    """Create a simple cube centered at the origin.

    Returns:
        NpFloatArray: A numpy array of shape (8, 3) representing the cube vertices.
    """
    return np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_basic_projection(simple_cube: NpFloatArray) -> None:
    """Test basic projection functionality, including output shape, padding, and centering.

    Args:
        simple_cube (NpFloatArray): Fixture providing cube coordinates.
    """
    projector = OrthographicProjection()
    params = OrthographicProjectionParameters(
        width=100, height=100, padding_x=0.05, padding_y=0.05, center=True
    )

    projected_data: NpFloatArray = projector.project(simple_cube, params)

    # Check output shape (N, 3) -> [X_canvas, Y_canvas, Depth]
    assert projected_data.shape == (8, 3)

    coords_2d = projected_data[:, :2]
    depth = projected_data[:, 2]

    # Check depth values (relative to default view [0, 0, 1])
    # Points with z=1 should have depth -1, points with z=-1 should have depth 1
    expected_depth = np.array([-(-1), -(1), -(-1), -(1), -(-1), -(1), -(-1), -(1)])
    assert np.allclose(depth, expected_depth)

    # Check bounds respect padding (canvas coords are [0, width] and [0, height])
    # With center=True, the range should fit within the available space
    available_width = 100 * (1 - 2 * 0.05)  # 90
    available_height = 100 * (1 - 2 * 0.05)  # 90
    canvas_center_x = 100 / 2
    canvas_center_y = 100 / 2

    min_x, min_y = np.min(coords_2d, axis=0)
    max_x, max_y = np.max(coords_2d, axis=0)

    # Check centering: mean should be close to canvas center
    assert np.allclose(
        np.mean(coords_2d, axis=0), [canvas_center_x, canvas_center_y], atol=1e-9
    )

    # Check that the *scaled range* fits within the available width/height
    # Since it's a cube projected orthographically, range_x = range_y = 2 * scale
    # scale = min(available_width / 2, available_height / 2) = 45
    assert np.allclose(max_x - min_x, available_width, atol=1e-9)
    assert np.allclose(max_y - min_y, available_height, atol=1e-9)

    # Check absolute bounds (min should be center - range/2, max should be center + range/2)
    assert np.allclose(min_x, canvas_center_x - available_width / 2)  # 50 - 45 = 5
    assert np.allclose(max_x, canvas_center_x + available_width / 2)  # 50 + 45 = 95
    assert np.allclose(min_y, canvas_center_y - available_height / 2)  # 50 - 45 = 5
    assert np.allclose(max_y, canvas_center_y + available_height / 2)  # 50 + 45 = 95


def test_view_directions() -> None:
    """Test that depth calculation correctly reflects the view direction."""
    projector = OrthographicProjection()
    coords = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=float
    )  # Points along z-axis

    # Front view (default: view=[0, 0, 1], up=[0, 1, 0])
    params = OrthographicProjectionParameters()
    projected_data_front: NpFloatArray = projector.project(coords, params)
    depth_front = projected_data_front[:, 2]
    # Depth = -(coords @ view) = -([0,0,1] @ [0,0,1]) = -1 for first point
    # Depth = -([0,0,-1] @ [0,0,1]) = 1 for second point
    assert np.allclose(depth_front, [-1.0, 1.0])

    # Back view (view=[0, 0, -1], up=[0, 1, 0])
    params.view_direction = np.array([0.0, 0.0, -1.0])
    projected_data_back: NpFloatArray = projector.project(coords, params)
    depth_back = projected_data_back[:, 2]
    # Depth = -(coords @ view) = -([0,0,1] @ [0,0,-1]) = 1 for first point
    # Depth = -([0,0,-1] @ [0,0,-1]) = -1 for second point
    assert np.allclose(depth_back, [1.0, -1.0])

    # Top view (view=[0, 1, 0], up=[0, 0, -1] -> needs adjustment for correct 'up')
    # View = [0, 1, 0]. Up should be roughly [-1, 0, 0] or [0, 0, -1]
    # Let's use up = [0, 0, -1]. Then right = cross(up, view) = cross([0,0,-1], [0,1,0]) = [1, 0, 0]
    params.view_direction = np.array([0.0, 1.0, 0.0])
    params.up_vector = np.array([0.0, 0.0, -1.0])  # Pointing down Z
    coords_plane = np.array(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
    )  # Points along y-axis
    projected_data_top: NpFloatArray = projector.project(coords_plane, params)
    depth_top = projected_data_top[:, 2]
    # Depth = -(coords @ view) = -([0,1,0] @ [0,1,0]) = -1
    # Depth = -([0,-1,0] @ [0,1,0]) = 1
    assert np.allclose(depth_top, [-1.0, 1.0])


def test_aspect_ratio_preservation() -> None:
    """Test that aspect ratio is maintained when requested."""
    projector = OrthographicProjection()
    # Rectangle in 3D space (width=2, height=1 -> aspect ratio 2:1)
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )

    # Canvas is wider (200x100)
    params_wide = OrthographicProjectionParameters(
        width=200, height=100, maintain_aspect_ratio=True, padding_x=0, padding_y=0
    )
    projected_wide: NpFloatArray = projector.project(coords, params_wide)
    coords_2d_wide = projected_wide[:, :2]
    width_wide = np.max(coords_2d_wide[:, 0]) - np.min(coords_2d_wide[:, 0])
    height_wide = np.max(coords_2d_wide[:, 1]) - np.min(coords_2d_wide[:, 1])
    # Height should be limiting factor. Scale = available_h / range_h = 100 / 1 = 100
    # Projected width = range_x * scale = 2 * 100 = 200
    # Projected height = range_y * scale = 1 * 100 = 100
    assert np.isclose(height_wide, 100.0)
    assert np.isclose(width_wide, 200.0)
    assert np.isclose(width_wide / height_wide, 2.0)

    # Canvas is taller (100x200)
    params_tall = OrthographicProjectionParameters(
        width=100, height=200, maintain_aspect_ratio=True, padding_x=0, padding_y=0
    )
    projected_tall: NpFloatArray = projector.project(coords, params_tall)
    coords_2d_tall = projected_tall[:, :2]
    width_tall = np.max(coords_2d_tall[:, 0]) - np.min(coords_2d_tall[:, 0])
    height_tall = np.max(coords_2d_tall[:, 1]) - np.min(coords_2d_tall[:, 1])
    # Width should be limiting factor. Scale = available_w / range_w = 100 / 2 = 50
    # Projected width = range_x * scale = 2 * 50 = 100
    # Projected height = range_y * scale = 1 * 50 = 50
    assert np.isclose(width_tall, 100.0)
    assert np.isclose(height_tall, 50.0)
    assert np.isclose(width_tall / height_tall, 2.0)


def test_centering_option() -> None:
    """Test the effect of the 'center' parameter."""
    projector = OrthographicProjection()
    # Use non-symmetrical coordinates to better show centering difference
    coords = np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 1.0]], dtype=float)
    # Input 2D projection will be [[1, 1], [1, 3]] (mean=[1, 2], range=[0, 2])
    width, height = 100, 100
    canvas_center = np.array([width / 2, height / 2])  # [50, 50]
    pad = 0.1
    padded_origin = np.array([width * pad, height * pad])  # [10, 10]
    padded_max_coord = np.array([width * (1 - pad), height * (1 - pad)])  # [90, 90]
    # ruff: noqa: F841
    available_width = width * (1 - 2 * pad)  # 80
    # ruff: noqa: F841
    available_height = height * (1 - 2 * pad)  # 80

    # Test with centering=True (use padding)
    params_centered = OrthographicProjectionParameters(
        width=width,
        height=height,
        canvas_alignment="center",
        center_original_coordinates=True,
        padding_x=pad,
        padding_y=pad,
    )
    projected_centered: NpFloatArray = projector.project(coords, params_centered)
    coords_2d_centered = projected_centered[:, :2]

    # Calculations for centered:
    # coords_2d_centered_view = [[0, -1], [0, 1]] (after subtracting mean [1, 2])
    # range = [0, 2]
    # scale = min(available_w/range_x, available_h/range_y) = min(80/0, 80/2) = 40
    # scaled = [[0*40, -1*40], [0*40, 1*40]] = [[0, -40], [0, 40]]
    # translated = scaled + canvas_center = [[0+50, -40+50], [0+50, 40+50]] = [[50, 10], [50, 90]]

    # Mean of projected coords should be canvas center
    assert np.allclose(np.mean(coords_2d_centered, axis=0), canvas_center, atol=1e-9)
    # Check exact coordinates
    expected_centered = np.array([[50.0, 10.0], [50.0, 90.0]])
    assert np.allclose(coords_2d_centered, expected_centered, atol=1e-9)

    # Test with centering=False (use padding)
    params_not_centered = OrthographicProjectionParameters(
        width=width, height=height, center=False, padding_x=pad, padding_y=pad
    )
    projected_not_centered: NpFloatArray = projector.project(
        coords, params_not_centered
    )
    coords_2d_not_centered = projected_not_centered[:, :2]

    # Calculations for not centered:
    # coords_2d_view = [[1, 1], [1, 3]]
    # range = [0, 2] (calculated based on coords_2d_view because center=False)
    # scale = min(80/0, 80/2) = 40
    # scaled = coords_2d_view * scale = [[1*40, 1*40], [1*40, 3*40]] = [[40, 40], [40, 120]]
    # scaled_min = [40, 40]
    # translated = scaled - scaled_min + padded_origin
    # translated = [[40-40+10, 40-40+10], [40-40+10, 120-40+10]] = [[10, 10], [10, 90]]

    # Mean should NOT be canvas center
    assert not np.allclose(
        np.mean(coords_2d_not_centered, axis=0), canvas_center, atol=1e-9
    )
    # Min of projected coords should be at padded origin
    assert np.allclose(np.min(coords_2d_not_centered, axis=0), padded_origin, atol=1e-9)
    # Max Y should be at padded_max_coord Y; Max X should match Min X (since input X was constant)
    assert np.allclose(
        np.max(coords_2d_not_centered, axis=0),
        [padded_origin[0], padded_max_coord[1]],
        atol=1e-9,
    )
    # Check exact values
    expected_not_centered = np.array([[10.0, 10.0], [10.0, 90.0]])
    assert np.allclose(coords_2d_not_centered, expected_not_centered, atol=1e-9)


def test_padding_behavior(simple_cube: NpFloatArray) -> None:
    """Test that padding is correctly applied when center=False.

    Args:
        simple_cube (NpFloatArray): Fixture providing cube coordinates.
    """
    projector = OrthographicProjection()

    test_cases = [
        (0.1, 0.2, 1000, 800),  # 10% x-padding, 20% y-padding
        (0.0, 0.0, 500, 500),  # No padding
        (0.25, 0.25, 400, 400),  # Equal 25% padding
    ]

    for pad_x, pad_y, width, height in test_cases:
        params = OrthographicProjectionParameters(
            width=width,
            height=height,
            padding_x=pad_x,
            padding_y=pad_y,
            maintain_aspect_ratio=False,  # Test padding independently
            center=False,  # Test non-centered padding behavior
        )

        projected_data: NpFloatArray = projector.project(simple_cube, params)
        coords_2d = projected_data[:, :2]

        # Calculate expected padded bounds
        expected_min_x = width * pad_x
        expected_max_x = width * (1 - pad_x)
        expected_min_y = height * pad_y
        expected_max_y = height * (1 - pad_y)

        # Check that coordinates are within padded bounds
        # With center=False, the min point should align with the padded origin
        assert np.allclose(np.min(coords_2d[:, 0]), expected_min_x, atol=1e-9)
        assert np.allclose(np.max(coords_2d[:, 0]), expected_max_x, atol=1e-9)
        assert np.allclose(np.min(coords_2d[:, 1]), expected_min_y, atol=1e-9)
        assert np.allclose(np.max(coords_2d[:, 1]), expected_max_y, atol=1e-9)


def test_padding_with_aspect_ratio(simple_cube: NpFloatArray) -> None:
    """Test padding behavior when maintain_aspect_ratio is True and center=True.

    Args:
        simple_cube (NpFloatArray): Fixture providing cube coordinates.
    """
    projector = OrthographicProjection()

    # Use the simple cube which projects to a square aspect ratio (1:1)
    params = OrthographicProjectionParameters(
        width=1000,  # Wider canvas
        height=500,  # Shorter canvas
        padding_x=0.1,  # 10% padding each side
        padding_y=0.1,  # 10% padding each side
        maintain_aspect_ratio=True,
        canvas_alignment="center",
        center_original_coordinates=True,
    )

    projected_data: NpFloatArray = projector.project(simple_cube, params)
    coords_2d = projected_data[:, :2]

    # Calculate usable canvas area after padding
    # ruff: noqa: F841
    usable_width = 1000 * (1 - 2 * 0.1)  # 800
    # ruff: noqa: F841
    usable_height = 500 * (1 - 2 * 0.1)  # 400

    # Calculate actual projection bounds (width and height of the projected shape)
    actual_width = np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0])
    actual_height = np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1])

    # Original aspect ratio is 1:1. Canvas aspect ratio is 1000:500 = 2:1.
    # Usable aspect ratio is 800:400 = 2:1.
    # Height is the limiting dimension for the 1:1 object.
    # Scale = min(usable_width / range_x, usable_height / range_y)
    # For the cube range_x = range_y = 2.
    # Scale = min(800 / 2, 400 / 2) = min(400, 200) = 200.
    # Expected actual width = range_x * scale = 2 * 200 = 400
    # Expected actual height = range_y * scale = 2 * 200 = 400

    assert np.allclose(actual_width, 400.0, atol=1e-9)
    assert np.allclose(
        actual_height, 400.0, atol=1e-9
    )  # Height is limited by usable_height
    assert np.allclose(
        actual_width / actual_height, 1.0, atol=1e-9
    )  # Aspect ratio maintained

    # Check centering within the canvas
    canvas_center_x = 1000 / 2
    canvas_center_y = 500 / 2
    assert np.allclose(
        np.mean(coords_2d, axis=0), [canvas_center_x, canvas_center_y], atol=1e-9
    )


def test_empty_input() -> None:
    """Test projection with empty input coordinates."""
    projector = OrthographicProjection()
    params = OrthographicProjectionParameters()
    empty_coords = np.empty((0, 3), dtype=float)

    projected_data: NpFloatArray = projector.project(empty_coords, params)

    assert projected_data.shape == (0, 3)
    assert projected_data.dtype == float


def test_invalid_input_shape() -> None:
    """Test projection with incorrectly shaped input coordinates."""
    projector = OrthographicProjection()
    params = OrthographicProjectionParameters()

    # Wrong number of dimensions
    invalid_coords_1d = np.array([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ValueError) as e:
        projector.project(invalid_coords_1d, params)  # type: ignore
    assert "Input coordinates must be a NumPy array with shape (N, 3)" in str(e.value)

    # Wrong number of columns
    invalid_coords_2col = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    with pytest.raises(ValueError) as e:
        projector.project(invalid_coords_2col, params)  # type: ignore
    assert "Input coordinates must be a NumPy array with shape (N, 3)" in str(e.value)

    # Non-numpy input
    invalid_coords_list = [[1.0, 2.0, 3.0]]
    with pytest.raises(ValueError) as e:
        projector.project(invalid_coords_list, params)  # type: ignore
    assert "Input coordinates must be a NumPy array with shape (N, 3)" in str(e.value)


def test_zero_vectors() -> None:
    """Test projection with zero view or up vectors."""
    projector = OrthographicProjection()
    coords = np.array([[1.0, 1.0, 1.0]], dtype=float)

    # Zero view vector
    params_zero_view = OrthographicProjectionParameters(
        view_direction=np.array([0.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="must be non-zero"):
        projector.project(coords, params_zero_view)

    # Zero up vector
    params_zero_up = OrthographicProjectionParameters(
        up_vector=np.array([0.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="must be non-zero"):
        projector.project(coords, params_zero_up)


def test_parallel_vectors(simple_cube: NpFloatArray) -> None:
    """Test projection when view and up vectors are parallel or anti-parallel.

    Args:
        simple_cube (NpFloatArray): Fixture providing cube coordinates.
    """
    projector = OrthographicProjection()

    # Parallel case
    params_parallel = OrthographicProjectionParameters(
        view_direction=np.array([0.0, 1.0, 0.0]),
        up_vector=np.array([0.0, 1.0, 0.0]),  # Same as view
    )
    # Should still work by picking a default orthogonal vector
    projected_parallel: NpFloatArray = projector.project(simple_cube, params_parallel)
    assert projected_parallel.shape == (8, 3)

    # Anti-parallel case
    params_anti = OrthographicProjectionParameters(
        view_direction=np.array([0.0, 1.0, 0.0]),
        up_vector=np.array([0.0, -1.0, 0.0]),  # Opposite of view
    )
    # Should also work
    projected_anti: NpFloatArray = projector.project(simple_cube, params_anti)
    assert projected_anti.shape == (8, 3)

    # Verify the depth is calculated correctly for the view=[0,1,0] case
    depth_parallel = projected_parallel[:, 2]
    # Depth = -(coords @ view) = -(coords @ [0,1,0]) = -y_coord
    expected_depth = -simple_cube[
        :, 1
    ]  # [-(-1), -(-1), -(1), -(1), -(-1), -(-1), -(1), -(1)]
    assert np.allclose(depth_parallel, expected_depth)

    # Depth should be the same for anti-parallel up vector case
    depth_anti = projected_anti[:, 2]
    assert np.allclose(depth_anti, expected_depth)


def test_single_point_projection(simple_cube: NpFloatArray) -> None:
    """Test projecting a single point, ensuring no division by zero errors.

    Args:
        simple_cube (NpFloatArray): Fixture providing cube coordinates (using first point).
    """
    projector = OrthographicProjection()
    single_point = simple_cube[0:1, :]  # Shape (1, 3)
    width, height = 200, 100
    pad_x, pad_y = 0.1, 0.1

    # Test centered
    params_centered = OrthographicProjectionParameters(
        width=width,
        height=height,
        padding_x=pad_x,
        padding_y=pad_y,
        canvas_alignment="center",
        center_original_coordinates=True,
    )
    projected_centered: NpFloatArray = projector.project(single_point, params_centered)
    assert projected_centered.shape == (1, 3)
    coords_2d_centered = projected_centered[:, :2]
    # Single point centered should be at the canvas center
    assert np.allclose(coords_2d_centered, [[width / 2, height / 2]], atol=1e-9)

    # Test not centered
    params_not_centered = OrthographicProjectionParameters(
        width=width, height=height, padding_x=pad_x, padding_y=pad_y, center=False
    )
    projected_not_centered: NpFloatArray = projector.project(
        single_point, params_not_centered
    )
    assert projected_not_centered.shape == (1, 3)
    coords_2d_not_centered = projected_not_centered[:, :2]
    # Single point not centered should be at the padded origin
    expected_origin = [width * pad_x, height * pad_y]  # [20, 10]
    assert np.allclose(coords_2d_not_centered, [expected_origin], atol=1e-9)
