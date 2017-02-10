# Name and path of resulting executable
Target = libText.so

MAKE ?= make
CXX ?= g++
CXXFLAGS +=  -g -ggdb -Wno-attributes -fPIC  `pkg-config --cflags  opencv`

OCR_LIBDIR = /opt/exe/PixelToText/OCR
OCR_LDFLAGS = -locr

CppSourceFiles = text_recongnition.cpp

ObjFiles = $(notdir $(CppSourceFiles:.cpp=.o))
LibIncString = $(addprefix -L, $(OCR_LIBDIR))
CXXFLAGS += $(LibIncString)

# Add -ldl to linker flags on Linux systems
ifeq ($(shell uname -s), Linux)
    dlLibName = -ldl
endif

LDFLAGS +=  $(OCR_LDFLAGS) -lboost_system -lboost_filesystem `pkg-config  --libs opencv` $(dlLibName) -Wl,-rpath-link $(FRE_LIBDIR)

.PHONY: all clean rebuild


all: $(Target)
	@echo Done

# Perform full rebuild of all binaries
rebuild:
	@$(MAKE)  --no-print-directory clean
	@$(MAKE)  --no-print-directory all

# Generate executable from object files
$(Target): $(ObjFiles)
	$(CXX) -shared -Wl,-soname,$(Target) $(CXXFLAGS) -o $(Target)  $^ $(LDFLAGS)

# Generate object files from .cpp files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm -f $(TargetPath) *.o *core *.so

