SRC=main.py
DIST=dist
EXE=HakaiKelpOMatic9000

all: $(DIST)/$(EXE)

$(DIST)/$(EXE): $(SRC)
	 pyinstaller -n $(EXE) --additional-hooks-dir=hooks --add-data form.ui:. --add-data models:. --add-data VERSION:. -p /home/taylor/PycharmProjects/uav-classif $(SRC)

clean:
	rm -rf dist build *.spec

.PHONY: all clean