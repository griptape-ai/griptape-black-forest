from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)


class TestBlackForestImageGenerationDriver:
    def test_init(self):
        assert BlackForestImageGenerationDriver(model="foo", api_key="bar")
