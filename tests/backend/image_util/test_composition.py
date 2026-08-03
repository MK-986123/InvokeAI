from io import BytesIO

from PIL import Image, ImageCms

from invokeai.backend.image_util.composition import apply_icc_profile, get_icc_profile


def test_get_icc_profile():
    # Test string profile names
    profile_srgb = get_icc_profile("sRGB")
    assert hasattr(profile_srgb, 'tobytes') or type(profile_srgb).__name__ == 'CmsProfile'

    profile_lab = get_icc_profile("LAB")
    assert hasattr(profile_lab, 'tobytes') or type(profile_lab).__name__ == 'CmsProfile'

    # Test PIL.ImageCms.ImageCmsProfile
    profile_same = get_icc_profile(profile_srgb)
    assert profile_same is profile_srgb

    # Test bytes
    profile_bytes = ImageCms.ImageCmsProfile(profile_srgb).tobytes()
    profile_from_bytes = get_icc_profile(profile_bytes)
    assert hasattr(profile_from_bytes, 'tobytes') or type(profile_from_bytes).__name__ == 'CmsProfile'

    # Test BytesIO
    profile_io = BytesIO(profile_bytes)
    profile_from_io = get_icc_profile(profile_io)
    assert hasattr(profile_from_io, 'tobytes') or type(profile_from_io).__name__ == 'CmsProfile'


def test_apply_icc_profile():
    img_rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 128))

    # Test RGBA to RGBA conversion using default sRGB
    out_rgba = apply_icc_profile(img_rgba, "sRGB", "sRGB", output_mode="RGBA")
    assert out_rgba.mode == "RGBA"
    assert out_rgba.size == (10, 10)

    # Test RGBA to LAB
    out_lab = apply_icc_profile(img_rgba, "sRGB", "LAB", output_mode="LAB")
    assert out_lab.mode == "LAB"
    assert out_lab.size == (10, 10)

    # Test RGB image to LAB
    img_rgb = Image.new("RGB", (10, 10), (255, 0, 0))
    out_rgb = apply_icc_profile(img_rgb, "sRGB", "LAB", output_mode="LAB")
    assert out_rgb.mode == "LAB"
    assert out_rgb.size == (10, 10)

    # Test applying profile to bytes
    profile_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    out_bytes_profile = apply_icc_profile(img_rgba, profile_bytes, "sRGB", output_mode="RGBA")
    assert out_bytes_profile.mode == "RGBA"
