编译
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=/tmp/t3/usr ..
	make 
	make install
	# 但是这里我把cmakelists.txt中的安装代码都注释掉了，避免对系统的干扰
运行
	没有可执行文件
