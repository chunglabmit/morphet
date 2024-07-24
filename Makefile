#-----------------------
# Makefile
#
# created on 08/15/2018
#-----------------------

include config.mk
TIMESTMP := $(shell /bin/date "+%Y%m%d-%H%M%S")
TIMESTMP_DATEONLY := $(shell /bin/date "+%Y%m%d")

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Requires [ $* ] to be set"; \
		exit 1; \
	fi

include makefiles/train_alnet.mk
include makefiles/gui.mk
include makefiles/data.mk

all:
	$(MAKE) usage.x

setup.env.x:
	bash shell/setup_env.sh

tensorboard.x:
	tensorboard --logdir ./src/train/logs --port 14234 serve

tensorboard.specific: guard-logdir guard-port
	tensorboard --logdir $(logdir) --port $(port) serve

notebook.x:
	jupyter notebook ./src/notebook/ --no-browser --port=$(JUPYTER_PORT)

notebook.lab:
	jupyter-lab ./src/notebook/

lab.x:
	cd ~/cbm/src/notebook/ && jupyter-lab --port=$(JUPYTER_PORT)

clean.x:
	rm `find . -name *.pyc`

test.x:
	pytest tests/

view.json.x: guard-paramf
	cat $(paramf) | python -m json.tool
