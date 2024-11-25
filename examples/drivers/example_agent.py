from griptape.black_forest.drivers.black_forest_image_generation_driver import (
    BlackForestImageGenerationDriver,
)
from griptape.engines import PromptImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import FileManagerTool, PromptImageGenerationTool

agent = Agent(
    tools=[
        PromptImageGenerationTool(
            engine=PromptImageGenerationEngine(
                image_generation_driver=BlackForestImageGenerationDriver(
                    model="flux-pro-1.1"
                )
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Save a cinematic, realistic picture of a dog riding a skateboard to the assets directory as dog_skateboard_cinematic.jpeg"
)
