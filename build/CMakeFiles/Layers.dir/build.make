# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.18

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "E:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "E:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Documents\Research\Thesis_work\Layers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Documents\Research\Thesis_work\build

# Include any dependencies generated for this target.
include CMakeFiles/Layers.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Layers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Layers.dir/flags.make

CMakeFiles/Layers.dir/layers.cpp.obj: CMakeFiles/Layers.dir/flags.make
CMakeFiles/Layers.dir/layers.cpp.obj: CMakeFiles/Layers.dir/includes_CXX.rsp
CMakeFiles/Layers.dir/layers.cpp.obj: D:/Documents/Research/Thesis_work/Layers/layers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Documents\Research\Thesis_work\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Layers.dir/layers.cpp.obj"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Layers.dir\layers.cpp.obj -c D:\Documents\Research\Thesis_work\Layers\layers.cpp

CMakeFiles/Layers.dir/layers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Layers.dir/layers.cpp.i"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Documents\Research\Thesis_work\Layers\layers.cpp > CMakeFiles\Layers.dir\layers.cpp.i

CMakeFiles/Layers.dir/layers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Layers.dir/layers.cpp.s"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Documents\Research\Thesis_work\Layers\layers.cpp -o CMakeFiles\Layers.dir\layers.cpp.s

CMakeFiles/Layers.dir/CNN.cpp.obj: CMakeFiles/Layers.dir/flags.make
CMakeFiles/Layers.dir/CNN.cpp.obj: CMakeFiles/Layers.dir/includes_CXX.rsp
CMakeFiles/Layers.dir/CNN.cpp.obj: D:/Documents/Research/Thesis_work/Layers/CNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Documents\Research\Thesis_work\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Layers.dir/CNN.cpp.obj"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Layers.dir\CNN.cpp.obj -c D:\Documents\Research\Thesis_work\Layers\CNN.cpp

CMakeFiles/Layers.dir/CNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Layers.dir/CNN.cpp.i"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Documents\Research\Thesis_work\Layers\CNN.cpp > CMakeFiles\Layers.dir\CNN.cpp.i

CMakeFiles/Layers.dir/CNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Layers.dir/CNN.cpp.s"
	E:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Documents\Research\Thesis_work\Layers\CNN.cpp -o CMakeFiles\Layers.dir\CNN.cpp.s

# Object files for target Layers
Layers_OBJECTS = \
"CMakeFiles/Layers.dir/layers.cpp.obj" \
"CMakeFiles/Layers.dir/CNN.cpp.obj"

# External object files for target Layers
Layers_EXTERNAL_OBJECTS =

Layers.exe: CMakeFiles/Layers.dir/layers.cpp.obj
Layers.exe: CMakeFiles/Layers.dir/CNN.cpp.obj
Layers.exe: CMakeFiles/Layers.dir/build.make
Layers.exe: CMakeFiles/Layers.dir/linklibs.rsp
Layers.exe: CMakeFiles/Layers.dir/objects1.rsp
Layers.exe: CMakeFiles/Layers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Documents\Research\Thesis_work\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Layers.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Layers.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Layers.dir/build: Layers.exe

.PHONY : CMakeFiles/Layers.dir/build

CMakeFiles/Layers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Layers.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Layers.dir/clean

CMakeFiles/Layers.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\Documents\Research\Thesis_work\Layers D:\Documents\Research\Thesis_work\Layers D:\Documents\Research\Thesis_work\build D:\Documents\Research\Thesis_work\build D:\Documents\Research\Thesis_work\build\CMakeFiles\Layers.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Layers.dir/depend

