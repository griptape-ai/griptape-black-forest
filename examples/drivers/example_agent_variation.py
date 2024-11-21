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
                    image_prompt_strength=1.0,
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Create a variation of assets/dog_skateboard_watercolor.jpeg, that looks like an anime puppy. Save it as dog_skateboard_anime.jpeg."
)
