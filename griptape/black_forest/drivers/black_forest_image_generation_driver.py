from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urljoin

import requests
from attrs import Factory, define, field

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageGenerationDriver


@define
class BlackForestImageGenerationDriver(BaseImageGenerationDriver):
    """Driver for the Black Forest Labs image generation API.

    Attributes:
        model: Black Forest model, for example 'flux-pro-1.1', 'flux-pro', 'flux-dev', 'flux-pro-1.1-ultra'.
        width: Width of the generated image. Valid for 'flux-pro-1.1', 'flux-pro', 'flux-dev' models only. Integer range from 256 to 1440. Must be a multiple of 32. Default is 1024.
        height: Height of the generated image. Valid for 'flux-pro-1.1', 'flux-pro', 'flux-dev' models only. Integer range from 256 to 1440. Must be a multiple of 32. Default is 1024.
        aspect_ratio: Aspect ratio of the generated image between 21:9 and 9:21. Valid for 'flux-pro-1.1-ultra' model only. Default is 16:9.
        prompt_upsampling: Optional flag to perform upsampling on the prompt. Valid for `flux-pro-1.1', 'flux-pro', 'flux-dev' models only. If active, automatically modifies the prompt for more creative generation.
        safety_tolerance: Optional tolerance level for input and output moderation. Valid for 'flux-pro-1.1', 'flux-pro', 'flux-dev' models only. Between 0 and 6, 0 being most strict, 6 being least strict.
        seed: Optional seed for reproducing results. Default is None.
        steps: Optional number of steps for the image generation process. Valid for 'flux-dev' model only. Default is None.


    """

    base_url: str = field(
        default="https://api.bfl.ml",
        kw_only=True,
        metadata={"serializable": False},
    )
    api_key: str | None = field(
        default=Factory(lambda: os.environ["BFL_API_KEY"]),
        kw_only=True,
        metadata={"serializable": False},
    )
    width: int = field(default=1024, kw_only=True)
    height: int = field(default=768, kw_only=True)
    sleep_interval: float = field(default=0.5, kw_only=True)
    safety_tolerance: int | None = field(default=None, kw_only=True)
    aspect_ratio: str = field(default=None, kw_only=True)
    seed: int | None = field(default=None, kw_only=True)
    prompt_upsampling: bool = field(default=False, kw_only=True)
    steps: int | None = field(default=None, kw_only=True)

    def try_text_to_image(
        self, prompts: list[str], negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        prompt = " ".join(prompts)

        if self.width % 32 != 0 or self.height % 32 != 0:
            msg = "width and height must be multiples of 32"
            raise ValueError(msg)
        if self.width < 256 or self.width > 1440:
            raise ValueError("width must be between 256 and 1440")
        if self.safety_tolerance and (
            self.safety_tolerance < 0 or self.safety_tolerance > 6
        ):
            raise ValueError("safety_tolerance must be between 0 and 6")

        data: dict[str, Any] = {
            "prompt": prompt,
        }

        if self.seed:
            data["seed"] = self.seed

        if self.model == "flux-pro-1.1-ultra" and self.aspect_ratio:
            data["aspect_ratio"] = self.aspect_ratio

        if self.model in ["flux-pro-1.1", "flux-pro", "flux-dev"]:
            data["width"] = self.width
            data["height"] = self.height
            if self.safety_tolerance:
                data["safety_tolerance"] = self.safety_tolerance
            if self.prompt_upsampling:
                data["prompt_upsampling"] = self.prompt_upsampling

        request = requests.post(
            urljoin(self.base_url, f"v1/{self.model}"),
            headers={
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=data,
        ).json()

        request_id = request["id"]

        image_url = None
        while True:
            time.sleep(self.sleep_interval)
            result = requests.get(
                urljoin(self.base_url, "v1/get_result"),
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                params={
                    "id": request_id,
                },
            ).json()
            if result["status"] == "Ready":
                image_url = result["result"]["sample"]
                break

        image_response = requests.get(image_url)
        image_bytes = image_response.content

        return ImageArtifact(
            value=image_bytes, format="jpeg", width=self.width, height=self.height
        )

    def try_image_variation(
        self,
        prompts: list[str],
        image: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support variation"
        )

    def try_image_inpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inpainting"
        )

    def try_image_outpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support outpainting"
        )
