SRC = main.py
EXE = HakaiKelpOMatic9000
MODELS = DeepLabV3_ResNet101_kelp_presence_jit.pt \
		 DeepLabV3_ResNet101_kelp_species_jit.pt \
		 LRASPP_MobileNetV3_kelp_presence_jit.pt \
		 LRASPP_MobileNetV3_kelp_species_jit.pt
MODELS_DIR = models
DIST_DIR = dist

.PHONY: all clean $(MODELS)

all: $(DIST_DIR)/$(EXE)

$(MODELS):
	mkdir -p ${MODELS_DIR}
	wget -c -N -O ${MODELS_DIR}/$@ https://hakai-deep-learning-datasets.s3.amazonaws.com/torchscripts/$@

$(DIST_DIR)/$(EXE): $(SRC) $(MODELS)
	 pyinstaller \
 		--noconfirm \
 		-n $(EXE) \
 		--additional-hooks-dir=hooks \
 		--add-data form.ui:. \
 		--add-data VERSION:. \
 		--add-data ${MODELS_DIR}:models \
 		-p /home/taylor/PycharmProjects/uav-classif \
 		$(SRC)
# 		--onefile \
# 		--console \

clean:
	rm -rf dist build *.spec
