# Griptape Black Forest Extension

## Overview
This extension provides an [Image Generation Driver](https://docs.griptape.ai/stable/griptape-framework/drivers/image-generation-drivers/#amazon-bedrock) for [Black Forest Labs](https://docs.bfl.ml/quick_start/gen_image).

```python
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
    "Save a picture of a watercolor painting of a dog riding a skateboard to the desktop."
)
```

## Installation

Poetry:
```bash
poetry add git+https://github.com/griptape-ai/griptape-black-forest.git
```

Pip:
```bash
pip install git+https://github.com/griptape-ai/griptape-black-forest.git
```
