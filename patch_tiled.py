with open("invokeai/app/invocations/tiled_multi_diffusion_denoise_latents.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "cfg_rescale_multiplier=self.cfg_rescale_multiplier," in line:
        continue
    new_lines.append(line)

with open("invokeai/app/invocations/tiled_multi_diffusion_denoise_latents.py", "w") as f:
    f.writelines(new_lines)
