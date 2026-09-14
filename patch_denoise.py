with open("invokeai/app/invocations/denoise_latents.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "cfg_rescale_multiplier: float," in line:
        continue
    if "cfg_rescale_multiplier=self.cfg_rescale_multiplier," in line:
        continue
    if "guidance_rescale_multiplier=cfg_rescale_multiplier," in line:
        continue
    new_lines.append(line)

with open("invokeai/app/invocations/denoise_latents.py", "w") as f:
    f.writelines(new_lines)
