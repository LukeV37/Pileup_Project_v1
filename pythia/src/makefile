SUBMODULES_PATH=../../submodules/

CPP=g++
LD=g++
SPECIALFLAGS=-O2
ROOTCFLAGS=$(shell root-config --cflags)
ROOTLIBS=$(shell root-config --libs)

CFLAGS = $(SPECIALFLAGS) -I.
LFLAGS = -L.

RCXX=$(CFLAGS) $(ROOTCFLAGS)
RLXX=$(LFLAGS) $(ROOTLIBS)

SRC1=generate_ttbar.cpp generate_ttbar_dict.cpp
SRC2=generate_dihiggs.cpp generate_dihiggs_dict.cpp

generate_ttbar_dict.cpp: ./include/LinkDef.h
	@rootcling generate_ttbar_dict.cpp ./include/LinkDef.h

generate_dihiggs_dict.cpp: ./include/LinkDef.h
	@rootcling generate_dihiggs_dict.cpp ./include/LinkDef.h

%.o: %.cpp
	@$(CPP) $(RCXX) -c $< -w -I $(SUBMODULES_PATH)/Pythia-8.312/include -O2 -std=c++17 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet 

generate_ttbar: $(SRC1:.cpp=.o)
	@$(LD) $(SRC1:.cpp=.o) $(RLXX) -o run_ttbar -w -I $(SUBMODULES_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet

generate_dihiggs: $(SRC2:.cpp=.o)
	@$(LD) $(SRC2:.cpp=.o) $(RLXX) -o run_dihiggs -w  -I $(SUBMODULES_PATH)/Pythia-8.312/include -O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread -L $(SUBMODULES_PATH)/Pythia-8.312/lib -Wl,-rpath,$(SUBMODULES_PATH)/Pythia-8.312/lib -lpythia8 -ldl -I$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/include -L$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -Wl,-rpath,$(SUBMODULES_PATH)/FastJet-3.4.2/fastjet-install/lib -lfastjet

clean:
	@rm -f *~ generate_*_dict* *.o *.exe
	@rm -f run_ttbar
	@rm -f run_dihiggs
	@rm -f run_hepmc
