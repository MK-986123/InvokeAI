# InvokeAI with Enhanced Flux Schedulers - Build Workflow

This repository contains InvokeAI with enhanced Flux scheduler support, providing multiple scheduler variants for improved image generation quality and control.

## Enhanced Flux Schedulers

This build includes 4 enhanced Flux schedulers:

1. **flow_euler** - Default FlowMatch Euler scheduler
   - Standard Euler integration method
   - Good balance of speed and quality

2. **flow_euler_k** - Euler with Karras sigmas
   - Uses Karras noise scheduling
   - Often provides better detail preservation

3. **flow_euler_exp** - Euler with exponential sigmas
   - Uses exponential noise scheduling  
   - Alternative scheduling approach for different aesthetic results

4. **flow_heun** - FlowMatch Heun scheduler
   - Higher-order Heun integration method
   - More accurate but slower than Euler methods

## GitHub Actions Workflow

The `.github/workflows/build-invokeai-flux.yml` workflow provides:

### Features
- **Multi-platform builds**: Linux x86-64 and Windows x86-64
- **Multi-Python support**: Python 3.11 and 3.12
- **Quality assurance**: Automated testing and validation
- **Artifact management**: Wheels and distribution packages
- **Release automation**: Optional release creation with manual trigger

### Workflow Jobs

1. **quality-checks**: Python formatting, linting, and Flux scheduler tests
2. **frontend-checks**: Frontend TypeScript validation and Flux integration checks  
3. **build-linux**: Build wheels for Linux x86-64
4. **build-windows**: Build wheels for Windows x86-64
5. **test-wheel**: Validate built wheels on both platforms
6. **create-release**: Create GitHub release with artifacts (manual trigger)

### Triggering the Workflow

#### Automatic Triggers
- Push to `main`, `release/**`, or `feature/**` branches
- Pull requests affecting Flux scheduler code
- Changes to build configuration files

#### Manual Triggers
- Workflow dispatch with options for:
  - Python version selection (3.10, 3.11, 3.12)
  - Release creation toggle

### Using the Built Packages

#### From Workflow Artifacts
1. Go to the Actions tab in GitHub
2. Select a successful workflow run
3. Download the appropriate artifact for your platform and Python version
4. Install the wheel:
   ```bash
   pip install InvokeAI-*.whl
   ```

#### From Releases (when created)
1. Go to the Releases page
2. Download the wheel for your platform and Python version
3. Install as above

### Verifying Flux Scheduler Installation

After installation, verify the enhanced schedulers are available:

```python
from invokeai.backend.flux.flux_schedulers import FLUX_SCHEDULER_MAP

print("Available Flux schedulers:")
for scheduler_name in FLUX_SCHEDULER_MAP.keys():
    print(f"  - {scheduler_name}")
```

Expected output:
```
Available Flux schedulers:
  - flow_euler
  - flow_euler_k  
  - flow_euler_exp
  - flow_heun
```

## Build Requirements

### System Requirements
- **Linux**: Ubuntu 20.04+ or compatible
- **Windows**: Windows 10+ with PowerShell/bash support
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: 20+ (for frontend building)
- **pnpm**: 10+ (package manager)

### Development Setup
```bash
# Clone the repository
git clone https://github.com/MK-986123/InvokeAI.git
cd InvokeAI

# Install development dependencies
pip install -e ".[dev,test]"

# Install frontend dependencies
cd invokeai/frontend/web
pnpm install
cd ../../..

# Run tests
python -m pytest tests/app/invocations/test_flux_denoise_schedulers.py -v
```

## Workflow Configuration

The workflow can be customized by modifying `.github/workflows/build-invokeai-flux.yml`:

### Key Configuration Points
- **Platform matrix**: Add/remove build platforms
- **Python versions**: Modify supported Python versions
- **Trigger conditions**: Customize when builds run
- **Artifact retention**: Adjust artifact storage duration
- **Test coverage**: Add/modify validation tests

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensure real-time output in logs
- `FORCE_COLOR=1`: Enable colored output in CI

## Architecture

```
InvokeAI with Enhanced Flux Schedulers
├── Enhanced Scheduler Implementation
│   ├── invokeai/backend/flux/flux_schedulers.py
│   ├── FLUX_SCHEDULER_MAP (scheduler classes)
│   └── FLUX_SCHEDULER_PARAMS (configuration)
├── Frontend Integration  
│   └── invokeai/frontend/web/src/features/nodes/util/graph/generation/buildFLUXGraph.ts
├── Testing Framework
│   └── tests/app/invocations/test_flux_denoise_schedulers.py
└── Build & Distribution
    ├── scripts/build_wheel.sh
    └── .github/workflows/build-invokeai-flux.yml
```

## Contributing

When contributing to the enhanced Flux schedulers:

1. **Modify scheduler code**: Update `invokeai/backend/flux/flux_schedulers.py`
2. **Add tests**: Update or add tests in `tests/app/invocations/`
3. **Update workflow**: Modify build workflow if needed
4. **Test locally**: Run the test suite before committing
5. **Create PR**: The workflow will automatically validate changes

## Support

For issues with the enhanced Flux schedulers or build process:

1. Check the [Issues](https://github.com/MK-986123/InvokeAI/issues) page
2. Run the workflow manually to test latest changes
3. Verify installation with the test script provided above

## License

This project follows the same license as the original InvokeAI project. See the `LICENSE` file for details.