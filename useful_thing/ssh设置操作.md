设置单独gitlab
1. 在.ssh文件夹生成rsa文件
ssh-keygen -t rsa -C "tianrnerui@lacesar.com:10443"	#默认名称id_rsa

2. 在gitlab上添加ssh公钥

3. ssh git@repo.lacesar.com
#成功

设置多个
1. 在.ssh文件夹生成rsa文件
ssh-keygen -t rsa -C "tianrnerui@lacesar.com:10443"	#端口有修改写在这里。回车后可以选择设置名称lacesar
ssh-keygen -t rsa -C "525845526@qq.com"		#设置名称mygit

2. 分别在两个网站添加ssh公钥

3. 在.ssh文件夹设置config文件
# lacesar_gitlab
Host repo.lacesar.com				#这个名字一般和下面HostName一样
HostName repo.lacesar.com				#github要用github.com，gitlab要用登陆后的网址
PreferredAuthentications publickey			#一般就是publickey
IdentityFile ~/.ssh/lacesar				#这里是路径和rsa文件的名称，就是上面设置的名称

# personal_github
Host github.com
HostName github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/mygit

4. 测试
ssh git@github.com				#这里@前面统一用git，后面用上面的Host
ssh git@repo.lacesar.com

