# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mlg/Documents/A-project/CPP/NeuralNetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mlg/Documents/A-project/CPP/NeuralNetwork

# Include any dependencies generated for this target.
include CMakeFiles/neuralnetwork.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neuralnetwork.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neuralnetwork.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neuralnetwork.dir/flags.make

CMakeFiles/neuralnetwork.dir/src/layer.cpp.o: CMakeFiles/neuralnetwork.dir/flags.make
CMakeFiles/neuralnetwork.dir/src/layer.cpp.o: src/layer.cpp
CMakeFiles/neuralnetwork.dir/src/layer.cpp.o: CMakeFiles/neuralnetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neuralnetwork.dir/src/layer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralnetwork.dir/src/layer.cpp.o -MF CMakeFiles/neuralnetwork.dir/src/layer.cpp.o.d -o CMakeFiles/neuralnetwork.dir/src/layer.cpp.o -c /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/layer.cpp

CMakeFiles/neuralnetwork.dir/src/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neuralnetwork.dir/src/layer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/layer.cpp > CMakeFiles/neuralnetwork.dir/src/layer.cpp.i

CMakeFiles/neuralnetwork.dir/src/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neuralnetwork.dir/src/layer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/layer.cpp -o CMakeFiles/neuralnetwork.dir/src/layer.cpp.s

CMakeFiles/neuralnetwork.dir/src/main.cpp.o: CMakeFiles/neuralnetwork.dir/flags.make
CMakeFiles/neuralnetwork.dir/src/main.cpp.o: src/main.cpp
CMakeFiles/neuralnetwork.dir/src/main.cpp.o: CMakeFiles/neuralnetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neuralnetwork.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralnetwork.dir/src/main.cpp.o -MF CMakeFiles/neuralnetwork.dir/src/main.cpp.o.d -o CMakeFiles/neuralnetwork.dir/src/main.cpp.o -c /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/main.cpp

CMakeFiles/neuralnetwork.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neuralnetwork.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/main.cpp > CMakeFiles/neuralnetwork.dir/src/main.cpp.i

CMakeFiles/neuralnetwork.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neuralnetwork.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/main.cpp -o CMakeFiles/neuralnetwork.dir/src/main.cpp.s

CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o: CMakeFiles/neuralnetwork.dir/flags.make
CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o: src/neural_network.cpp
CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o: CMakeFiles/neuralnetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o -MF CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o.d -o CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o -c /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neural_network.cpp

CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neural_network.cpp > CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.i

CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neural_network.cpp -o CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.s

CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o: CMakeFiles/neuralnetwork.dir/flags.make
CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o: src/neuron.cpp
CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o: CMakeFiles/neuralnetwork.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o -MF CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o.d -o CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o -c /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neuron.cpp

CMakeFiles/neuralnetwork.dir/src/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neuralnetwork.dir/src/neuron.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neuron.cpp > CMakeFiles/neuralnetwork.dir/src/neuron.cpp.i

CMakeFiles/neuralnetwork.dir/src/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neuralnetwork.dir/src/neuron.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mlg/Documents/A-project/CPP/NeuralNetwork/src/neuron.cpp -o CMakeFiles/neuralnetwork.dir/src/neuron.cpp.s

# Object files for target neuralnetwork
neuralnetwork_OBJECTS = \
"CMakeFiles/neuralnetwork.dir/src/layer.cpp.o" \
"CMakeFiles/neuralnetwork.dir/src/main.cpp.o" \
"CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o" \
"CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o"

# External object files for target neuralnetwork
neuralnetwork_EXTERNAL_OBJECTS =

neuralnetwork: CMakeFiles/neuralnetwork.dir/src/layer.cpp.o
neuralnetwork: CMakeFiles/neuralnetwork.dir/src/main.cpp.o
neuralnetwork: CMakeFiles/neuralnetwork.dir/src/neural_network.cpp.o
neuralnetwork: CMakeFiles/neuralnetwork.dir/src/neuron.cpp.o
neuralnetwork: CMakeFiles/neuralnetwork.dir/build.make
neuralnetwork: CMakeFiles/neuralnetwork.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable neuralnetwork"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neuralnetwork.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neuralnetwork.dir/build: neuralnetwork
.PHONY : CMakeFiles/neuralnetwork.dir/build

CMakeFiles/neuralnetwork.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neuralnetwork.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neuralnetwork.dir/clean

CMakeFiles/neuralnetwork.dir/depend:
	cd /Users/mlg/Documents/A-project/CPP/NeuralNetwork && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mlg/Documents/A-project/CPP/NeuralNetwork /Users/mlg/Documents/A-project/CPP/NeuralNetwork /Users/mlg/Documents/A-project/CPP/NeuralNetwork /Users/mlg/Documents/A-project/CPP/NeuralNetwork /Users/mlg/Documents/A-project/CPP/NeuralNetwork/CMakeFiles/neuralnetwork.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/neuralnetwork.dir/depend

