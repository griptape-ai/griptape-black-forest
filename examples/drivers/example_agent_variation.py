from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.engines import VariationImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import (
    FileManagerTool,
    VariationImageGenerationTool,
)

agent = Agent(
    tools=[
        VariationImageGenerationTool(
            engine=VariationImageGenerationEngine(
                image_generation_driver=BlackForestImageGenerationDriver(
                    model="flux-pro-1.1-ultra",  # flux-pro-1.1-ultra is a better model for image variation
                    image_prompt_strength=0.7,
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "dog, skateboard, pencil sketch, black and white. Image file: assets/dog_skateboard_cinematic.jpeg. Save it to the assets directory as dog_skateboard_sketch.jpeg."
)
