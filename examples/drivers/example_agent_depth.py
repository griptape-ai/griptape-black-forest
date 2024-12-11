from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.structures import Agent
from griptape.tools import FileManagerTool, VariationImageGenerationTool

agent = Agent(
    tools=[
        VariationImageGenerationTool(
            image_generation_driver=BlackForestImageGenerationDriver(
                model="flux-pro-1.0-depth",  # flux-pro-1.0-depth is required for Depth.
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Set in a jungle, dog rides on a skateboard, use assets/dog_skateboard_cinematic.jpeg as the control image, and write it to assets/dog_skateboard_depth.jpeg "
)
