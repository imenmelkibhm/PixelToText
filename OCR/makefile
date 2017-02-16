# (c) 2013 ABBYY Production LLC 
# Description: FineReader Engine 11.1 makefile for samples

# Name and path of resulting executable
TargetPath = ./OCR

MAKE ?= make
CXX ?= g++
CXXFLAGS +=  -g -ggdb -Wno-attributes -fPIC

FRE_LIBDIR = /opt/exe/FREngine11/Bin
FRE_INCDIR = /opt/exe/FREngine11/Inc
FRE_LDFLAGS = -lFREngine

CppSourceFiles = OCR.cpp $(wildcard /opt/exe/FREngine11/Samples/SharedTools/*.cpp)

ObjFiles = $(notdir $(CppSourceFiles:.cpp=.o))
IncludeString = $(addprefix -I, $(FRE_INCDIR)) -I/opt/exe/FREngine11/Samples/SharedTools
LibIncString = $(addprefix -L, $(FRE_LIBDIR))
CXXFLAGS += $(IncludeString) $(LibIncString)

# Add -ldl to linker flags on Linux systems
ifeq ($(shell uname -s), Linux)
    dlLibName = -ldl
endif

LDFLAGS +=  $(FRE_LDFLAGS) -lpthread $(dlLibName) -Wl,-rpath-link $(FRE_LIBDIR)

.PHONY: all clean rebuild

VPATH = /opt/exe/FREngine11/Samples/SharedTools

all: $(TargetPath)
	@echo Done

# Perform full rebuild of all binaries
rebuild:
	@$(MAKE)  --no-print-directory clean
	@$(MAKE)  --no-print-directory all

# Generate .so from object files
$(TargetPath): $(ObjFiles)
	$(CXX) -shared -Wl,-soname,libocr.so $(CXXFLAGS) -o libocr.so $^ $(LDFLAGS)

# Generate object files from .cpp files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm -f $(TargetPath) *.o *core *.so

# Generate executable from object files
ocr: $(ObjFiles)
	$(CXX) $(CXXFLAGS) -o ocr.exe $^ $(LDFLAGS)