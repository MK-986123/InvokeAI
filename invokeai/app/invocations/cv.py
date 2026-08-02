# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


import cv2 as cv
import numpy
from PIL import ImageOps

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.util import cv2_to_pil, pil_to_cv2


@invocation("cv_inpaint", title="OpenCV Inpaint", tags=["opencv", "inpaint"], category="inpaint", version="1.3.1")
class CvInpaintInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Simple inpaint using opencv."""

    image: ImageField = InputField(description="The image to inpaint")
    mask: ImageField = InputField(description="The mask to use when inpainting")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mask = context.images.get_pil(self.mask.image_name)

        # Convert to cv image/mask
        cv_image = pil_to_cv2(image)
        cv_mask = numpy.array(ImageOps.invert(mask.convert("L")))

        # Inpaint
        cv_inpainted = cv.inpaint(cv_image, cv_mask, 3, cv.INPAINT_TELEA)

        # Convert back to Pillow
        image_inpainted = cv2_to_pil(cv_inpainted)

        image_dto = context.images.save(image=image_inpainted)

        return ImageOutput.build(image_dto)
