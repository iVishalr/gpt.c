CC ?= gcc
CFLAGS = -O3 -Ofast -march=native -Wno-unused-result -ggdb3 -fPIC -fopenmp -DOMP
# CFLAGS = -Wno-unused-result -O0 -ggdb3 -fPIC -fopenmp -DOMP
INCLUDES = -I include/ -I third_party/OpenBLAS/include/
LDLIBS = -lm -lopenblas -lgomp
LDFLAGS = -L third_party/OpenBLAS/lib/

SRC_DIR = src
INCLUDE_DIR = include
BUILD = build
LIB_NAME=gpt
SHARED_SUFFIX=dll
PLATFORM=$(shell uname -s)
ifeq "$(PLATFORM)" "Darwin"
    SHARED_SUFFIX=dylib
endif
ifeq "$(PLATFORM)" "Linux"
    SHARED_SUFFIX=so
endif
SHARED_LIB = lib$(LIB_NAME).$(SHARED_SUFFIX)
# Check if Valgrind is installed
VALGRIND := $(shell command -v valgrind)

# Find all source files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names from source file names
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD)/%.o, $(SRCS))

# Find all C files in the project root directory
ROOT_SRCS = $(wildcard ./*.c)

# Generate executable names from root source file names
EXES = $(patsubst ./%.c, ./%, $(ROOT_SRCS))

.PHONY: all clean third_party shared_lib root_binaries setup valgrind_inference tests

# Default rule to build the shared library and root binaries
all: setup third_party shared_lib root_binaries

# Compile rule for object files
$(BUILD)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to create the shared library
$(SHARED_LIB): $(OBJS)
	$(CC) -o $(SHARED_LIB) $(LDFLAGS) $(LDLIBS) $^ -shared

# Compile rule for root source files into executables
$(EXES): $(ROOT_SRCS) $(SHARED_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.c -L . -l$(LIB_NAME) $(LDLIBS)

# Rule to build third_party libraries
third_party:
	@if ! test -d third_party/OpenBLAS/lib;\
		then cd third_party/OpenBLAS && make -j && make PREFIX=../../third_party/OpenBLAS install;\
	fi
# Rule to build the shared library
shared_lib: $(SHARED_LIB)

# Rule to build the root binaries
root_binaries: $(EXES)

tests: 
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) tests/test_bmm.c $(LDLIBS)

valgrind_inference: setup all
ifeq "$(VALGRIND)" ""
	@echo "Valgrind is not installed on your system!"
ifeq "$(PLATFORM)" "Darwin"
	@echo "If you are using brew, valgrind can be installed using 'brew install valgrind'"
	@exit 1
endif
ifeq "$(PLATFORM)" "Linux"
	@echo "Valgrind can be installed using 'sudo apt install valgrind'"
	@exit 1
endif
endif
	@if ! test -f "model/valgrind-model.bin"; then \
		python3 model.py --block-size=128 --layers=1 --heads=1 --embd=128 --name=valgrind-model; \
	fi
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./infer_gpt --load-checkpoint=model/valgrind-model.bin --tokenizer=tokenizer.bin --prompt="[50256, 17250]" --max-tokens=10

# Rule to create the build directory if it doesn't exist
setup: third_party
	@if ! test -d $(BUILD);\
		then echo "\033[93mSetting up build directory...\033[0m"; mkdir -p build;\
	fi

# Clean rule to remove all generated files
clean:
	rm -rf $(BUILD) a.out $(SHARED_LIB) $(EXES)
