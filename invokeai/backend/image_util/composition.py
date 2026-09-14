# TODO: Improve blend modes
# TODO: Add nodes like Hue Adjust for Saturation/Contrast/etc... ?
# TODO: Continue implementing more blend modes/color spaces(?)
# TODO: Blend multiple layers all crammed into a tensor(?) or list

# Copyright (c) 2023 Darren Ringer <dwringer@gmail.com>
# Parts based on Oklab: Copyright (c) 2021 Bj�rn Ottosson <https://bottosson.github.io/>
# HSL code based on CPython: Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved
from io import BytesIO
from math import pi as PI
from pathlib import Path
from typing import Union

import torch
from PIL import Image, ImageCms

from invokeai.backend.image_util.color_conversion import (
    gamut_clip_tensor,
)
from invokeai.backend.image_util.color_conversion import (
    srgb_from_linear_srgb as shared_srgb_from_linear_srgb,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

MAX_FLOAT = torch.finfo(torch.tensor(1.0).dtype).max


def get_icc_profile(
    profile: Union[str, Path, BytesIO, bytes, ImageCms.ImageCmsProfile],
) -> ImageCms.ImageCmsProfile:
    """Helper to get an ImageCmsProfile from various types."""
    if isinstance(profile, (ImageCms.ImageCmsProfile, ImageCms.core.CmsProfile)):
        return profile
    elif isinstance(profile, bytes):
        return ImageCms.getOpenProfile(BytesIO(profile))
    elif isinstance(profile, (str, Path)):
        if str(profile).lower() == "srgb":
            return ImageCms.createProfile("sRGB")
        elif str(profile).lower() == "lab":
            return ImageCms.createProfile("LAB")
        elif str(profile).lower() == "xyz":
            return ImageCms.createProfile("XYZ")
        return ImageCms.getOpenProfile(str(profile))
    elif hasattr(profile, "read"):
        if hasattr(profile, "seek"):
            profile.seek(0)
        return ImageCms.getOpenProfile(profile)
    else:
        raise TypeError(f"Invalid type for profile: {type(profile)}")


def apply_icc_profile(
    image: Image.Image,
    input_profile: Union[str, Path, BytesIO, bytes, ImageCms.ImageCmsProfile],
    output_profile: Union[str, Path, BytesIO, bytes, ImageCms.ImageCmsProfile] = "sRGB",
    rendering_intent: int = ImageCms.Intent.PERCEPTUAL,
    output_mode: Union[str, None] = None,
) -> Image.Image:
    """
    Applies an ICC profile to an image.
    """
    in_profile = get_icc_profile(input_profile)
    out_profile = get_icc_profile(output_profile)

    has_alpha = "A" in image.mode
    alpha = None
    if has_alpha:
        alpha = image.getchannel("A")
        image = image.convert("RGB")
        if output_mode is None:
            output_mode = "RGBA"

    transform_mode = output_mode
    if has_alpha and output_mode == "RGBA":
        transform_mode = "RGB"
    elif output_mode is None:
        transform_mode = image.mode

    transformed_image = ImageCms.profileToProfile(
        image,
        in_profile,
        out_profile,
        renderingIntent=rendering_intent,
        outputMode=transform_mode,
    )

    if transformed_image is None:
        transformed_image = image.copy()

    if has_alpha and output_mode == "RGBA":
        transformed_image = transformed_image.convert("RGBA")
        transformed_image.putalpha(alpha)

    return transformed_image


# CIE Lab to Uniform Perceptual Lab profile is copyright © 2003 Bruce Justin Lindbloom. All rights reserved. <http://www.brucelindbloom.com>
CIELAB_TO_UPLAB_ICC_PATH = Path(__file__).parent / "assets" / "CIELab_to_UPLab.icc"


def equivalent_achromatic_lightness(lch_tensor: torch.Tensor):
    """Calculate Equivalent Achromatic Lightness accounting for Helmholtz-Kohlrausch effect"""
    # As described by High, Green, and Nussbaum (2023): https://doi.org/10.1002/col.22839

    k = [0.1644, 0.0603, 0.1307, 0.0060]

    h_minus_90 = torch.sub(lch_tensor[2, :, :], PI / 2.0)
    h_minus_90 = torch.sub(torch.remainder(torch.add(h_minus_90, 3 * PI), 2 * PI), PI)

    f_by = torch.add(k[0] * torch.abs(torch.sin(torch.div(h_minus_90, 2.0))), k[1])
    f_r_0 = torch.add(k[2] * torch.abs(torch.cos(lch_tensor[2, :, :])), k[3])

    f_r = torch.zeros(lch_tensor[0, :, :].shape)
    mask_hi = torch.ge(lch_tensor[2, :, :], -1 * (PI / 2.0))
    mask_lo = torch.le(lch_tensor[2, :, :], PI / 2.0)
    mask = torch.logical_and(mask_hi, mask_lo)
    f_r[mask] = f_r_0[mask]

    l_max = torch.ones(lch_tensor[0, :, :].shape)
    l_min = torch.zeros(lch_tensor[0, :, :].shape)
    l_adjustment = torch.tensordot(torch.add(f_by, f_r), lch_tensor[1, :, :], dims=([0, 1], [0, 1]))
    l_max = torch.add(l_max, l_adjustment)
    l_min = torch.add(l_min, l_adjustment)
    l_eal_tensor = torch.add(lch_tensor[0, :, :], l_adjustment)

    l_eal_tensor = torch.add(
        lch_tensor[0, :, :], torch.tensordot(torch.add(f_by, f_r), lch_tensor[1, :, :], dims=([0, 1], [0, 1]))
    )
    l_eal_tensor = torch.div(torch.sub(l_eal_tensor, l_min.min()), l_max.max() - l_min.min())

    return l_eal_tensor


def srgb_from_linear_srgb(linear_srgb_tensor: torch.Tensor, alpha: float = 0.0, steps: int = 1):
    """Get gamma-corrected sRGB from a linear-light sRGB image tensor"""

    if 0.0 < alpha:
        linear_srgb_tensor = gamut_clip_tensor(linear_srgb_tensor, alpha=alpha, steps=steps)
    return shared_srgb_from_linear_srgb(linear_srgb_tensor)


def remove_nans(tensor: torch.Tensor, replace_with: float = MAX_FLOAT):
    return torch.where(torch.isnan(tensor), replace_with, tensor)


def tensor_from_pil_image(img: Image.Image, normalize: bool = False):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)


# PSF LICENSE AGREEMENT FOR PYTHON 3.11.5

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
#    the Individual or Organization ("Licensee") accessing and otherwise using Python
#    3.11.5 software in source or binary form and its associated documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
#    grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
#    analyze, test, perform and/or display publicly, prepare derivative works,
#    distribute, and otherwise use Python 3.11.5 alone or in any derivative
#    version, provided, however, that PSF's License Agreement and PSF's notice of
#    copyright, i.e., "Copyright (c) 2001-2023 Python Software Foundation; All Rights
#    Reserved" are retained in Python 3.11.5 alone or in any derivative version
#    prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
#    incorporates Python 3.11.5 or any part thereof, and wants to make the
#    derivative work available to others as provided herein, then Licensee hereby
#    agrees to include in any such work a brief summary of the changes made to Python
#    3.11.5.

# 4. PSF is making Python 3.11.5 available to Licensee on an "AS IS" basis.
#    PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
#    EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
#    WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
#    USE OF PYTHON 3.11.5 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.11.5
#    FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
#    MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.11.5, OR ANY DERIVATIVE
#    THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material breach of
#    its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any relationship
#    of agency, partnership, or joint venture between PSF and Licensee.  This License
#    Agreement does not grant permission to use PSF trademarks or trade name in a
#    trademark sense to endorse or promote products or services of Licensee, or any
#    third party.

# 8. By copying, installing or otherwise using Python 3.11.5, Licensee agrees
#    to be bound by the terms and conditions of this License Agreement.
######################################################################################/
