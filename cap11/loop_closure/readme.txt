编译
	mkdir build
	cd build
	cmake ..
	make
运行
	./loop_closure ../src/vocabulary.yml.gz
	或者
	./loop_closure ../src/vocab_larger.yml.gz

	vocabulary.yml.gz和vocab_larger.yml.gz来自于feature_training和gen_vocab_large的运行产物
