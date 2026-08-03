import timeit

setup = """
from pathlib import Path
import os
filename = 'test/image_123.jpg'
"""

path_code = """
basename = Path(filename).name
"""

os_path_code = """
basename = os.path.basename(filename)
"""

print(f"Path(filename).name: {timeit.timeit(path_code, setup=setup, number=1000000)} seconds")
print(f"os.path.basename(filename): {timeit.timeit(os_path_code, setup=setup, number=1000000)} seconds")
