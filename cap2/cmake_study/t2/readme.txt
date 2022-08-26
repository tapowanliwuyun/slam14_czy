编译
	mkdir build
	cd build
	cmake ..
	make 
	
	//如果想把文件安装到制定的位置，就把注释去掉，然后使用以下命令编译
	cmake -DCMAKE_INSTALL_PREFIX=/tmp/t2/usr ..
	make
	make install
	
运行
	这个文件不生成可执行文件
