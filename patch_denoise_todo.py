with open("invokeai/app/invocations/denoise_latents.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "TODO: old backend, remove" in line:
        continue
    new_lines.append(line)

with open("invokeai/app/invocations/denoise_latents.py", "w") as f:
    f.writelines(new_lines)
