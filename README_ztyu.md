原始版本：
2021年11月11日, 948048b39986b7c585c0f8e9ae6bfebc89073636.

目的:
1. 增加高斯八叉树模块
2. 增加高斯八叉树测试模块 和 bin2octomap 211204

注意:
1. 若出现 " undefined symbol: _ZTIN7octomap14GaussionOcTree "
  删除ros-melodic-octomap。 ros自带库lib文件，没有Gaussion相关信息