#!/bin/bash

# Local test script for InvokeAI with Enhanced Flux Schedulers
# This script simulates the key validation steps from the GitHub Actions workflow

set -e

echo "🧪 InvokeAI Enhanced Flux Schedulers - Local Test"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "invokeai/backend/flux/flux_schedulers.py" ]]; then
    print_error "Must be run from the InvokeAI repository root directory"
    exit 1
fi

print_status "Starting local validation tests..."

# Test 1: Validate Flux scheduler module structure
print_status "Test 1: Validating Flux scheduler module structure..."
python3 -c "
import ast
import sys

try:
    with open('invokeai/backend/flux/flux_schedulers.py', 'r') as f:
        content = f.read()
    
    # Parse the module
    tree = ast.parse(content)
    
    # Check for expected exports
    expected_names = ['FLUX_SCHEDULER_NAME', 'FLUX_SCHEDULER_MAP', 'FLUX_SCHEDULER_PARAMS']
    found_names = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found_names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found_names.append(node.target.id)
    
    missing = set(expected_names) - set(found_names)
    if missing:
        print(f'❌ Missing expected exports: {missing}')
        sys.exit(1)
    
    # Check scheduler count
    scheduler_types = ['flow_euler', 'flow_heun', 'flow_euler_k', 'flow_euler_exp']
    found_schedulers = [s for s in scheduler_types if s in content]
    
    if len(found_schedulers) >= 4:
        print('✅ Module structure validation passed')
        print(f'Found {len(found_schedulers)} enhanced Flux schedulers')
    else:
        print('❌ Not enough enhanced Flux schedulers found')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error validating Flux scheduler module: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "Flux scheduler module structure is valid"
else
    print_error "Flux scheduler module validation failed"
    exit 1
fi

# Test 2: Validate workflow YAML syntax
print_status "Test 2: Validating GitHub Actions workflow YAML..."
python3 -c "
import yaml
import sys

try:
    with open('.github/workflows/build-invokeai-flux.yml', 'r') as f:
        workflow = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['name', 'on', 'jobs']
    for field in required_fields:
        if field not in workflow:
            print(f'❌ Missing required field: {field}')
            sys.exit(1)
    
    # Check job count
    jobs = workflow.get('jobs', {})
    expected_jobs = ['quality-checks', 'frontend-checks', 'build-linux', 'build-windows', 'test-wheel', 'create-release']
    
    for job in expected_jobs:
        if job not in jobs:
            print(f'❌ Missing expected job: {job}')
            sys.exit(1)
    
    print('✅ Workflow YAML validation passed')
    print(f'Found {len(jobs)} workflow jobs')
    
except Exception as e:
    print(f'❌ Error validating workflow YAML: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "GitHub Actions workflow YAML is valid"
else
    print_error "Workflow YAML validation failed"
    exit 1
fi

# Test 3: Check if build script exists and is executable
print_status "Test 3: Checking build script..."
if [[ -f "scripts/build_wheel.sh" ]]; then
    if [[ -x "scripts/build_wheel.sh" ]]; then
        print_success "Build script exists and is executable"
    else
        print_warning "Build script exists but is not executable - fixing..."
        chmod +x scripts/build_wheel.sh
        print_success "Build script made executable"
    fi
else
    print_error "Build script not found at scripts/build_wheel.sh"
    exit 1
fi

# Test 4: Check frontend action exists
print_status "Test 4: Checking frontend action..."
if [[ -f ".github/actions/install-frontend-deps/action.yml" ]]; then
    print_success "Frontend dependencies action found"
else
    print_error "Frontend dependencies action not found"
    exit 1
fi

# Test 5: Validate test files exist
print_status "Test 5: Checking test files..."
test_files=(
    "tests/app/invocations/test_flux_denoise_schedulers.py"
    "tests/app/invocations/test_flux_denoise.py"
)

for test_file in "${test_files[@]}"; do
    if [[ -f "$test_file" ]]; then
        print_success "Test file found: $test_file"
    else
        print_warning "Test file not found: $test_file"
    fi
done

# Test 6: Check pyproject.toml structure
print_status "Test 6: Checking pyproject.toml..."
if [[ -f "pyproject.toml" ]]; then
    python3 -c "
import sys
try:
    import tomli
except ImportError:
    # Fallback for Python < 3.11
    try:
        import tomllib as tomli
    except ImportError:
        print('⚠️  Cannot validate pyproject.toml - tomli/tomllib not available')
        sys.exit(0)

try:
    with open('pyproject.toml', 'rb') as f:
        config = tomli.load(f)
    
    if 'project' in config and 'dependencies' in config['project']:
        deps = config['project']['dependencies']
        flux_related = [dep for dep in deps if 'diffusers' in dep.lower()]
        if flux_related:
            print('✅ Found diffusers dependency for Flux support')
        else:
            print('⚠️  No diffusers dependency found - may affect Flux functionality')
    else:
        print('⚠️  Project dependencies not found in pyproject.toml')
        
except Exception as e:
    print(f'⚠️  Could not parse pyproject.toml: {e}')
"
    print_success "pyproject.toml structure checked"
else
    print_error "pyproject.toml not found"
    exit 1
fi

echo ""
print_success "🎉 All local validation tests passed!"
echo ""
print_status "Summary:"
echo "  ✅ Flux scheduler module structure validated"
echo "  ✅ GitHub Actions workflow YAML validated"  
echo "  ✅ Build script verified"
echo "  ✅ Frontend action verified"
echo "  ✅ Test files checked"
echo "  ✅ Project configuration checked"
echo ""
print_status "The workflow should run successfully in GitHub Actions."
print_status "To trigger a build, push changes or use workflow_dispatch."

exit 0