#!/bin/bash
# Script to run integration smoke tests locally

set -e  # Exit on error

echo "========================================="
echo "VideoLingo Integration Smoke Tests"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if pytest is installed
if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${YELLOW}Warning: pytest not found. Installing...${NC}"
    pip install pytest pytest-mock pytest-timeout pytest-cov
fi

# Check if FFmpeg is available (optional but recommended)
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg is available${NC}"
else
    echo -e "${YELLOW}⚠ FFmpeg not found. Some tests may use mock video generation${NC}"
fi

# Check if numpy is installed (required for synthetic video generation)
if ! python3 -c "import numpy" &> /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: numpy not found. Installing...${NC}"
    pip install numpy
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Parse command line arguments
RUN_MODE="smoke"  # Default to smoke tests
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            RUN_MODE="full"
            shift
            ;;
        --performance)
            RUN_MODE="performance"
            shift
            ;;
        --verbose|-v)
            VERBOSE="-vv"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=core --cov-report=term-missing --cov-report=html"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --full         Run full integration test suite"
            echo "  --performance  Run performance tests"
            echo "  --verbose, -v  Verbose output"
            echo "  --coverage     Generate coverage report"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Default: Run smoke tests only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create test results directory
mkdir -p test_results

echo ""
echo "Configuration:"
echo "  Mode: $RUN_MODE"
echo "  Python: $(python3 --version)"
echo "  Pytest: $(python3 -m pytest --version | head -n 1)"
echo ""

# Run tests based on mode
case $RUN_MODE in
    smoke)
        echo "Running integration smoke tests..."
        echo "================================="
        python3 -m pytest \
            tests/integration/test_smoke_integration.py \
            -m "integration and slow" \
            $VERBOSE \
            --tb=short \
            --timeout=300 \
            $COVERAGE \
            --junit-xml=test_results/smoke_tests.xml \
            2>&1 | tee test_results/smoke_tests.log
        ;;
        
    full)
        echo "Running full integration test suite..."
        echo "======================================"
        python3 -m pytest \
            tests/integration \
            tests/e2e \
            -m "integration" \
            $VERBOSE \
            --tb=short \
            --timeout=600 \
            $COVERAGE \
            --junit-xml=test_results/full_tests.xml \
            2>&1 | tee test_results/full_tests.log
        ;;
        
    performance)
        echo "Running performance integration tests..."
        echo "========================================"
        # Install performance testing dependencies if needed
        if ! python3 -c "import psutil" &> /dev/null 2>&1; then
            echo -e "${YELLOW}Installing performance testing dependencies...${NC}"
            pip install psutil memory_profiler
        fi
        
        python3 -m pytest \
            tests/integration \
            -m "integration and performance" \
            $VERBOSE \
            --tb=short \
            --timeout=900 \
            --junit-xml=test_results/performance_tests.xml \
            2>&1 | tee test_results/performance_tests.log
        ;;
esac

# Check test results
TEST_EXIT_CODE=$?

echo ""
echo "========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
    
    # Display coverage report if generated
    if [ -n "$COVERAGE" ]; then
        echo ""
        echo "Coverage report generated:"
        echo "  - Terminal: See above"
        echo "  - HTML: htmlcov/index.html"
    fi
else
    echo -e "${RED}✗ Some tests failed. Check the logs for details.${NC}"
    echo "  Log file: test_results/${RUN_MODE}_tests.log"
fi

echo "========================================="

# Display test results location
echo ""
echo "Test results saved to: test_results/"
ls -la test_results/

exit $TEST_EXIT_CODE
