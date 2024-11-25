from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.engines import VariationImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import FileManagerTool, VariationImageGenerationTool

agent = Agent(
    tools=[
        VariationImageGenerationTool(
            engine=VariationImageGenerationEngine(
                image_generation_driver=BlackForestImageGenerationDriver(
                    model="flux-pro-1.0-canny",  # flux-pro-1.0-canny is required for Canny.
                    guidance=100,
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Childrens messy crayon drawing of a dog on a skateboard, use assets/dog_skateboard_cinematic.jpeg as the control image, and write it to assets/dog_skateboard_canny.jpeg "
)
