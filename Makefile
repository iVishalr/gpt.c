# CC=gcc
# CPP=g++
# CFLAGS=-O3 -Ofast -Wno-unused-result 
# INCLUDES=-I include/ -I /opt/OpenBLAS/include/
# LDLIBS=-lm -lopenblas
# LDFLAGS=-L /opt/OpenBLAS/lib

# SRC=src
# BUILD=build
# CSRCS=$(wildcard $(SRC)/*.c)
# OBJS=$(patsubst $(SRC)/%.c, $(BUILD)/%.o, $(CSRCS))

# .PHONY: all gpt setup clean

# all: setup gpt

# setup:
# 	@if ! test -d $(BUILD);\
# 		then echo "\033[93mSetting up build directory...\033[0m"; mkdir -p build;\
# 	fi

# gpt: $(OBJS)
# 	$(CC) $(OBJS) $(LDFLAGS) $(LDLIBS)

# $(OBJS): $(CSRCS)
# 	echo "$(CSRCS)"
# 	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -c $< -o $@ $(LDLIBS)

# clean:
# 	rm -rf $(BUILD) a.out

CC = gcc
CFLAGS = -O2 -Ofast -march=native -Wno-unused-result -ggdb3
# CFLAGS = -Wno-unused-result -O0 -ggdb3
INCLUDES = -I include/ -I /opt/OpenBLAS/include/
LDLIBS = -lm -lopenblas
LDFLAGS = -L /opt/OpenBLAS/lib

SRC_DIR = src
INCLUDE_DIR = include
BUILD = build
TARGET = $(BUILD)/gpt

# Find all source files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names from source file names
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD)/%.o, $(SRCS))

.PHONY: all clean gpt tests valgrind

# Default rule to build the target
all: setup gpt 

# Compile rule for object files
$(BUILD)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -c $< -o $@ $(LDLIBS)

# Compile rule for the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LDLIBS)

gpt: $(TARGET)

tests: 
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) tests/test_bmm.c $(LDLIBS)

valgrind: setup gpt
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./build/gpt 

# Rule to create the build directory if it doesn't exist
setup:
	@if ! test -d $(BUILD);\
		then echo "\033[93mSetting up build directory...\033[0m"; mkdir -p build;\
	fi

# Clean rule to remove all generated files
clean:
	rm -rf $(BUILD) a.out
