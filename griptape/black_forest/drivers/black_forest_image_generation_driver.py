from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urljoin

import requests
from attrs import Factory, define, field

from griptape.artifacts import ImageArtifact
from griptape.drivers import BaseImageGenerationDriver
from griptape.utils import import_optional_dependency


def steps_validator(instance, attribute, value):
    if value and (value < 1 or value > 50):
        raise ValueError("steps must be between 1 and 50")


def size_validator(instance, attribute, value):
    if value and value % 32 != 0:
        raise ValueError(f"{attribute} must be a multiple of 32")
    if value and value < 256 or value > 1440:
        raise ValueError(f"{attribute} must be between 256 and 1440")


def safety_validator(instance, attribute, value):
    if value and (value < 0 or value > 6):
        raise ValueError("safety_tolerance must be between 0 and 6")


def aspect_ratio_validator(instance, attribute, value):
    if value:
        width, height = value.split(":")
        if int(width) < 9 or int(width) > 21 or int(height) < 9 or int(height) > 21:
            raise ValueError("aspect_ratio must be between 9:21 and 21:9")


def guidance_validator(instance, attribute, value):
    if value and (value < 1.5 or value > 5):
        raise ValueError("guidance must be between 1.5 and 5")


def interval_validator(instance, attribute, value):
    if value and (value < 1 or value > 4):
        raise ValueError("interval must be between 1 and 4")


def image_prompt_strength_validator(instance, attribute, value):
    if value and (value < 0 or value > 1):
        raise ValueError("image_prompt_strength must be between 0 and 1")


@define
class BlackForestImageGenerationDriver(BaseImageGenerationDriver):
    """Driver for the Black Forest Labs image generation API.

    Attributes:
        model: Black Forest model, for example 'flux-pro-1.1', 'flux-pro', 'flux-dev', 'flux-pro-1.1-ultra'. Note - for inpainting, use 'flux-pro-1.0'.
        width: Width of the generated image. Valid for 'flux-pro-1.1', 'flux-pro', 'flux-dev' models only. Integer range from 256 to 1440. Must be a multiple of 32. Default is 1024.
        height: Height of the generated image. Valid for 'flux-pro-1.1', 'flux-pro', 'flux-dev' models only. Integer range from 256 to 1440. Must be a multiple of 32. Default is 1024.
        aspect_ratio: Aspect ratio of the generated image between 21:9 and 9:21. Valid for 'flux-pro-1.1-ultra' model only. Default is 16:9.
        prompt_upsampling: Optional flag to perform upsampling on the prompt. Valid for `flux-pro-1.1', 'flux-pro', 'flux-dev' models only. If active, automatically modifies the prompt for more creative generation.
        safety_tolerance: Optional tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict.
        seed: Optional seed for reproducing results. Default is None.
        steps: Optional number of steps for the image generation process. Valid for 'flux-dev' and `flux-pro` models only. Can be a value between 1 and 50. Default is None.
        guidance: Optional guidance scale for image generation. High guidance scales improve prompt adherence at the cost of reduced realism. Min: 1.5, max: 5. Valid for 'flux-dev' and 'flux-pro' models only.
        interval: Optional interval parameter for guidance control. Valid for 'flux-pro' model only. Value is an integer between 1 and 4. Default is None.
        raw: Optional flag to generate less processed, more natural-looking images. Valid for 'flux-pro-1.1-ultra' model only. Default is False.
        output_format: Output format of the generated image. Can be 'jpeg' or 'png'. Default is 'jpeg'.
        image_prompt_strength: Optional flag to control the strength of the image prompt. Valid for 'flux-pro-1.1-ultra' model only. Valid is float between 0 and 1. Default is 0.1.
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
    width: int = field(default=1024, kw_only=True, validator=size_validator)
    height: int = field(default=768, kw_only=True, validator=size_validator)
    sleep_interval: float = field(default=0.5, kw_only=True)
    safety_tolerance: int | None = field(
        default=None, kw_only=True, validator=safety_validator
    )
    aspect_ratio: str = field(
        default=None, kw_only=True, validator=aspect_ratio_validator
    )
    seed: int | None = field(default=None, kw_only=True)
    prompt_upsampling: bool = field(default=False, kw_only=True)
    steps: int | None = field(default=None, kw_only=True, validator=steps_validator)
    guidance: float | None = field(
        default=None, kw_only=True, validator=guidance_validator
    )
    interval: int | None = field(
        default=None, kw_only=True, validator=interval_validator
    )
    raw: bool = field(default=False, kw_only=True)
    output_format: str = field(default="jpeg", kw_only=True)
    image_prompt_strength: float = field(default=0.1, kw_only=True)

    def try_text_to_image(
        self, prompts: list[str], negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        prompt = " ".join(prompts)

        data: dict[str, Any] = {
            "prompt": prompt,
        }

        if self.seed:
            data["seed"] = self.seed
        if self.safety_tolerance:
            data["safety_tolerance"] = self.safety_tolerance

        if self.aspect_ratio and self.model == "flux-pro-1.1-ultra":
            data["aspect_ratio"] = self.aspect_ratio
            data["image_prompt_strength"] = self.image_prompt_strength

        if self.raw and self.model == "flux-pro-1.1-ultra":
            data["raw"] = self.raw

        if self.guidance and self.model in ["flux-dev", "flux-pro"]:
            data["guidance"] = float(self.guidance)

        if self.steps and self.model in ["flux-dev", "flux-pro"]:
            data["steps"] = int(self.steps)

        if self.interval and self.model == "flux-pro":
            data["interval"] = int(self.interval)

        if self.model in ["flux-pro-1.1", "flux-pro", "flux-dev"]:
            data["width"] = self.width
            data["height"] = self.height
            if self.prompt_upsampling:
                data["prompt_upsampling"] = self.prompt_upsampling
        if self.output_format:
            data["output_format"] = self.output_format
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
            value=image_bytes,
            format=self.output_format,
            width=self.width,
            height=self.height,
        )

    def try_image_variation(
        self,
        prompts: list[str],
        image: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        prompt = " ".join(prompts)

        # Get the base64 encoded image data
        image_data = image.base64

        # Confirm the image_data is indeed base64 encoded
        if not self._is_base64(image_data):
            raise ValueError("Image data is not base64 encoded.")

        data: dict[str, Any] = {
            "prompt": prompt,
        }
        if image_data:
            data["image_prompt"] = image_data
        if self.seed:
            data["seed"] = self.seed
        if self.safety_tolerance:
            data["safety_tolerance"] = self.safety_tolerance

        if self.aspect_ratio and self.model == "flux-pro-1.1-ultra":
            data["aspect_ratio"] = self.aspect_ratio
            data["image_prompt_strength"] = self.image_prompt_strength

        if self.raw and self.model == "flux-pro-1.1-ultra":
            data["raw"] = self.raw

        if self.guidance and self.model in ["flux-dev", "flux-pro"]:
            data["guidance"] = float(self.guidance)

        if self.steps and self.model in ["flux-dev", "flux-pro"]:
            data["steps"] = int(self.steps)

        if self.interval and self.model == "flux-pro":
            data["interval"] = int(self.interval)

        if self.model in ["flux-pro-1.1", "flux-pro", "flux-dev"]:
            data["width"] = self.width
            data["height"] = self.height
            if self.prompt_upsampling:
                data["prompt_upsampling"] = self.prompt_upsampling
        if self.output_format:
            data["output_format"] = self.output_format
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
            value=image_bytes,
            format=self.output_format,
            width=self.width,
            height=self.height,
        )

    def try_image_inpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        prompt = " ".join(prompts)

        # Get the base64 encoded image data
        image_data = image.base64

        # Get the mask data
        mask_data = mask.base64

        # Confirm the image_data is indeed base64 encoded
        if not self._is_base64(image_data):
            raise ValueError("Image data is not base64 encoded.")

        data: dict[str, Any] = {
            "prompt": prompt,
        }
        if image_data:
            data["image"] = image_data
        if mask_data:
            data["mask"] = mask_data

        if self.seed:
            data["seed"] = self.seed
        if self.safety_tolerance:
            data["safety_tolerance"] = self.safety_tolerance

        if self.aspect_ratio and self.model == "flux-pro-1.1-ultra":
            data["aspect_ratio"] = self.aspect_ratio
            data["image_prompt_strength"] = self.image_prompt_strength

        if self.raw and self.model == "flux-pro-1.1-ultra":
            data["raw"] = self.raw

        if self.guidance and self.model in ["flux-dev", "flux-pro"]:
            data["guidance"] = float(self.guidance)

        if self.steps and self.model in ["flux-dev", "flux-pro"]:
            data["steps"] = int(self.steps)

        if self.interval and self.model == "flux-pro":
            data["interval"] = int(self.interval)

        if self.model in ["flux-pro-1.1", "flux-pro", "flux-dev"]:
            data["width"] = self.width
            data["height"] = self.height
            if self.prompt_upsampling:
                data["prompt_upsampling"] = self.prompt_upsampling
        if self.output_format:
            data["output_format"] = self.output_format
        request = requests.post(
            urljoin(self.base_url, f"v1/{self.model}-fill"),
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
            value=image_bytes,
            format=self.output_format,
            width=self.width,
            height=self.height,
        )

    def try_image_outpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        # outpainting is done using the same method as inpainting
        artifact = self.try_image_inpainting(prompts, image, mask, negative_prompts)
        return artifact

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support outpainting"
        )

    def _is_base64(self, s: str) -> bool:
        base64 = import_optional_dependency("base64")

        if len(s) % 4 != 0:
            return False

        try:
            # Decode and then re-encode to check if it matches the original
            return base64.b64encode(base64.b64decode(s)).decode("utf-8") == s
        except Exception:
            return False
