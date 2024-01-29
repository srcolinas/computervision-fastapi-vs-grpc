import functools
from typing import Annotated, Callable, Iterable, cast

import cv2
import fastapi
import pydantic_settings
import numpy as np
import numpy.typing as npt

app = fastapi.FastAPI()


class Settings(pydantic_settings.BaseSettings):
    num_classes: int = 2
    num_outputs: int = 1

    def __hash__(self) -> int:
        return hash((self.num_classes, self.num_outputs))

    class Config:
        env_file = ".env"


@functools.lru_cache
def get_settings():
    return Settings()


Distribution = list[float]
MultiOutputPrediction = list[Distribution]
NumpyImage = Annotated[npt.NDArray[np.uint8], "shape=(height,width,channels)"]
Model = Callable[[Iterable[NumpyImage]], Iterable[MultiOutputPrediction]]


@functools.lru_cache
def get_model(settings: Settings = fastapi.Depends(get_settings)) -> Model:
    def helper(images: Iterable[NumpyImage]) -> Iterable[MultiOutputPrediction]:
        for _ in images:
            yield [[0.5] * settings.num_classes] * settings.num_outputs

    return helper


@app.post("/predict", response_model=list[MultiOutputPrediction])
def predict(files: list[fastapi.UploadFile], model: Model = fastapi.Depends(get_model)):
    """
    For each image uploaded, you get a distribution over classes for each
    output.

    """
    images = _decode(files)
    return model(images)


def _decode(files: Iterable[fastapi.UploadFile]) -> Iterable[NumpyImage]:
    for file in files:
        content = file.file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cast(NumpyImage, cv2.imdecode(nparr, cv2.IMREAD_COLOR))
        yield image
