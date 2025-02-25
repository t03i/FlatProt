import numpy as np
import pytest
from flatprot.projection.orthographic import (
    OrthographicProjector,
    OrthographicProjectionParameters,
)


@pytest.fixture
def simple_cube():
    """Create a simple cube centered at origin."""
    return np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )


def test_basic_projection(simple_cube):
    projector = OrthographicProjector()
    params = OrthographicProjectionParameters(
        width=100, height=100, padding_x=0.05, padding_y=0.05
    )

    coords_2d, depth = projector.project(simple_cube, params)

    # Check output shapes
    assert coords_2d.shape == (8, 2)
    assert depth.shape == (8,)

    # Check bounds respect padding
    assert np.all(coords_2d[:, 0] >= -45)  # width * padding_x = 100 * 0.05 = 5
    assert np.all(coords_2d[:, 0] <= 45)  # width * (1 - padding_x) = 100 * 0.95 = 95
    assert np.all(coords_2d[:, 1] >= -45)
    assert np.all(coords_2d[:, 1] <= 45)

    # Check centering
    assert np.allclose(np.mean(coords_2d, axis=0), [0, 0], atol=1e-10)


def test_view_directions():
    projector = OrthographicProjector()
    coords = np.array([[0, 0, 1], [0, 0, -1]])  # Points along z-axis

    # Front view
    params = OrthographicProjectionParameters(
        view_direction=np.array([0, 0, 1]), up_vector=np.array([0, 1, 0])
    )
    coords_2d, depth = projector.project(coords, params)
    assert np.allclose(depth, [-1, 1])  # Point at -1 is closer in front view

    # Back view
    params.view_direction = np.array([0, 0, -1])
    coords_2d, depth = projector.project(coords, params)
    assert np.allclose(depth, [1, -1])  # Point ordering reversed


def test_aspect_ratio_preservation():
    projector = OrthographicProjector()
    # Rectangle in 3D space
    coords = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]])

    params = OrthographicProjectionParameters(
        width=200, height=100, maintain_aspect_ratio=True
    )

    coords_2d, _ = projector.project(coords, params)
    # Calculate aspect ratio of projected coordinates
    width = np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0])
    height = np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1])
    projected_ratio = width / height

    # Original aspect ratio was 2:1
    assert np.allclose(projected_ratio, 2.0)


def test_centering_option():
    projector = OrthographicProjector()
    # Offset coordinates
    coords = np.array([[1, 1, 1], [2, 2, 2]])

    # Test with centering
    params = OrthographicProjectionParameters(center=True)
    coords_2d, _ = projector.project(coords, params)
    assert np.allclose(np.mean(coords_2d, axis=0), [0, 0], atol=1e-10)

    # Test without centering
    params.center = False
    coords_2d, _ = projector.project(coords, params)
    assert not np.allclose(np.mean(coords_2d, axis=0), [0, 0])


def test_padding_behavior(simple_cube):
    """Test that padding is correctly applied as percentages of canvas dimensions."""
    projector = OrthographicProjector()

    # Test different padding configurations
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
            maintain_aspect_ratio=False,  # To test padding independently
            center=False,
        )

        coords_2d, _ = projector.project(simple_cube, params)

        # Check that coordinates are within padded bounds
        assert np.all(coords_2d[:, 0] >= width * pad_x)
        assert np.all(coords_2d[:, 0] <= width * (1 - pad_x))
        assert np.all(coords_2d[:, 1] >= height * pad_y)
        assert np.all(coords_2d[:, 1] <= height * (1 - pad_y))


def test_padding_with_aspect_ratio():
    """Test padding behavior when maintain_aspect_ratio is True."""
    projector = OrthographicProjector()

    # Rectangle in 3D space (2:1 aspect ratio)
    coords = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]])

    params = OrthographicProjectionParameters(
        width=1000, height=500, padding_x=0.1, padding_y=0.1, maintain_aspect_ratio=True
    )

    coords_2d, _ = projector.project(coords, params)

    # Calculate usable canvas area
    usable_width = 1000 * (1 - 2 * 0.1)  # 80% of width
    usable_height = 500 * (1 - 2 * 0.1)  # 80% of height

    # Calculate actual projection bounds
    actual_width = np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0])
    actual_height = np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1])

    # The smaller dimension should be fully utilized (minus padding)
    # while maintaining the 2:1 aspect ratio
    assert np.allclose(actual_height, usable_height, rtol=1e-10)
    assert np.allclose(actual_width / actual_height, 2.0, rtol=1e-10)
