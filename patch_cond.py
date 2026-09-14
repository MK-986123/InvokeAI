with open("invokeai/backend/stable_diffusion/diffusion/conditioning_data.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "guidance_rescale_multiplier: float = 0" in line:
        continue
    if "TODO: old backend, remove" in line and not "guidance_rescale_multiplier: float" in line:
        continue
    if 'For models trained using zero-terminal SNR ("ztsnr"), it\'s suggested to use guidance_rescale_multiplier of 0.7.' in line:
        continue
    if "See [Common Diffusion Noise Schedules and Sample Steps are Flawed]" in line:
        continue
    if "self.guidance_rescale_multiplier = guidance_rescale_multiplier" in line:
        continue
    new_lines.append(line)

with open("invokeai/backend/stable_diffusion/diffusion/conditioning_data.py", "w") as f:
    f.writelines(new_lines)
