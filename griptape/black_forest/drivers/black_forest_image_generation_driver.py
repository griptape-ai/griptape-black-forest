from __future__ import annotations
import os
import time

from urllib.parse import urljoin
from griptape.artifacts import ImageArtifact
import requests

from attrs import define, field, Factory

from griptape.drivers import BaseImageGenerationDriver


@define
class BlackForestImageGenerationDriver(BaseImageGenerationDriver):
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

    def try_text_to_image(
        self, prompts: list[str], negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        prompt = " ".join(prompts)

        request = requests.post(
            urljoin(self.base_url, f"v1/{self.model}"),
            headers={
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "width": self.width,
                "height": self.height,
            },
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
