#-----------------------
# gui.mk
#
# created on 03/18/2019
#-----------------------

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Requires [ $* ] to be set"; \
		exit 1; \
	fi

launch.ng.viewer.x:
	python -i src/ng/launch_viewer.py

launch.evaluator.x:
	cd src/gui/evaluator && python main.py train_val

launch.evaluator.remote:
	cd src/gui/evaluator && vglrun python main.py train_val

convert.ui2py.x:
	cd src/gui && bash convert_ui_to_py.sh annotation_tool.ui annotation_tool_ui.py

launch.morphet.x:
	cd src/gui/MorPheT && python main.py
