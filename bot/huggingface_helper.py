import logging
import os
import random
from time import sleep
from urllib.parse import urljoin

import requests


class HuggingFaceClient:
    def __init__(self, token: str, text_to_image_model_id: str = "stabilityai/stable-diffusion-2-1", retries: int = 3):
        if not token:
            raise ValueError("Token is required")
        self._api_url = "https://api-inference.huggingface.co/models/"
        self._token = token
        self._text_to_image_model_id = text_to_image_model_id
        self._retries = max(1, retries)

    def _get_model_url(self, model_id: str):
        return urljoin(self._api_url, model_id)

    def _make_request(self, url, payload):
        headers = {"Authorization": f"Bearer {self._token}"}
        n = 0
        while n < self._retries:
            response = requests.post(url, json=payload, headers=headers)
            if response == 503:
                logging.info("Got 503, retrying in 1-5 seconds")
                sleep(random.randint(1, 5))
                n += 1
                continue
            else:
                break
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.content.decode()}")
        return response

    async def _query(self, model_id: str, payload: dict):
        api_url = self._get_model_url(model_id)
        response = self._make_request(api_url, payload)
        return response.content

    async def text_to_image(self, prompt: str):
        payload = {"inputs": prompt}
        return await self._query(self._text_to_image_model_id, payload)


async def main():
    """Use this function to test the api client.
    """
    from PIL import Image
    import io

    client = HuggingFaceClient(os.getenv("HUGGINGFACE_TOKEN"),
                               "stabilityai/stable-diffusion-2-base")
    img_bytes = await client.text_to_image("A hacker coding at night.")
    img = Image.open(io.BytesIO(img_bytes))
    img.save("test.png")
    print("Image saved to test.png")


if __name__ == '__main__':

    import asyncio

    asyncio.run(main())
