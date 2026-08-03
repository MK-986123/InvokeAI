with open("invokeai/backend/stable_diffusion/diffusers_pipeline.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier" in line:
        continue
    if "if guidance_rescale_multiplier > 0:" in line:
        skip = True
        continue
    if skip and line.strip() == ")":
        skip = False
        continue
    if skip:
        continue

    if "@staticmethod" in line and i+1 < len(lines) and "def _rescale_cfg" in lines[i+1]:
        skip = True
        continue
    if "def _rescale_cfg" in line:
        skip = True
        continue
    if skip and "return x_final" in line:
        skip = False
        continue
    if skip:
        continue

    new_lines.append(line)

with open("invokeai/backend/stable_diffusion/diffusers_pipeline.py", "w") as f:
    f.writelines(new_lines)
