from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.engines import OutpaintingImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import FileManagerTool, OutpaintingImageGenerationTool

agent = Agent(
    tools=[
        OutpaintingImageGenerationTool(
            engine=OutpaintingImageGenerationEngine(
                image_generation_driver=BlackForestImageGenerationDriver(
                    model="flux-pro-1.0",  # flux-pro-1.0 is required for outpainting.
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Replace the environment in assets/dog_skateboard_cinematic.jpeg, with boat at sea, using the assets/dog_skateboard_env_mask.jpeg. Save it as dog_skateboard_at_sea.jpeg."
)
