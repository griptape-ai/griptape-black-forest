from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.engines import InpaintingImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import FileManagerTool, InpaintingImageGenerationTool

agent = Agent(
    tools=[
        InpaintingImageGenerationTool(
            engine=InpaintingImageGenerationEngine(
                image_generation_driver=BlackForestImageGenerationDriver(
                    model="flux-pro-1.0",  # flux-pro-1.0 is required for inpainting.
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Replace the dog in assets/dog_skateboard_watercolor.jpeg, with a walrus, using the assets/dog_skateboard_mask.jpeg. Save it as walrus_skateboard.jpeg."
)
