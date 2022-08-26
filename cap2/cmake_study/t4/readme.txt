编译
	mkdir build
	cd build
	cmake ..
	make # 由于该项目是对t3项目安装到本地的一种检验，所以如果t3没有安装到本地，那么自然没有指明目标并且找不到 makefile。 停止。
	ldd src/main # 用于检查可执行文件链接库的情况
	
运行
	./main
