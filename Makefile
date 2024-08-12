# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17

# Source files
SRCS = test_math.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Header files
HEADERS = math.h

# Output executable
TARGET = test_project

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
    $(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile source files into object files
%.o: %.cpp $(HEADERS)
    $(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
    rm -f $(OBJS) $(TARGET)

# Run tests
test: $(TARGET)
    ./$(TARGET)

.PHONY: all clean test