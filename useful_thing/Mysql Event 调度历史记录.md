目前的mysql版本没有Event执行历史信息，为方便查看Event是否正常执行以及执行结果，可以通过以下两个步骤来实现：

一．创建作业执行Event历史记录表

`CREATE TABLE `mysql`.`t_event_history` (
  `dbname` VARCHAR(128) NOT NULL DEFAULT '',
  `eventname` VARCHAR(128) NOT NULL DEFAULT '',
  `starttime` DATETIME NOT NULL DEFAULT '0000-00-00 00:00:00',
  `endtime` DATETIME DEFAULT NULL,
  `issuccess` INT(11) DEFAULT NULL,
  `duration` INT(11) DEFAULT NULL,
  `errormessage` VARCHAR(512) DEFAULT NULL,
  `randno` INT(11) DEFAULT NULL,
  PRIMARY KEY (`dbname`,`eventname`,`starttime`),
  KEY `ix_endtime` (`endtime`),
  KEY `ix_starttime_randno` (`starttime`,`randno`)
) ENGINE=INNODB DEFAULT CHARSET=utf8;`

二．根据以下建模板创建作业

#请注意根据实际情况修改相关信息

`DELIMITER $$
CREATE DEFINER=`root`@`localhost` EVENT `e_test` ON SCHEDULE 
#修改以下调度信息
EVERY 1 DAY STARTS '2014-01-03 01:00:00' ON COMPLETION PRESERVE ENABLE DO 
BEGIN
	DECLARE r_code CHAR(5) DEFAULT '00000';
	DECLARE r_msg TEXT;
	DECLARE v_error INTEGER;
	DECLARE	v_starttime DATETIME DEFAULT NOW();
	DECLARE v_randno INTEGER DEFAULT FLOOR(RAND()*100001);
	
	INSERT INTO mysql.t_event_history (dbname,eventname,starttime,randno) 
	#修改下面的作业名（该作业的名称）
	VALUES(DATABASE(),'e_test', v_starttime,v_randno);	
	
	BEGIN
		#异常处理段
		DECLARE CONTINUE HANDLER FOR SQLEXCEPTION  
		BEGIN
			SET  v_error = 1;
			GET DIAGNOSTICS CONDITION 1 r_code = RETURNED_SQLSTATE , r_msg = MESSAGE_TEXT;
		END;
		
		#此处为实际调用的用户程序过程
		CALL test.usp_test1();
	END;
	
	UPDATE mysql.t_event_history SET endtime=NOW(),issuccess=ISNULL(v_error),duration=TIMESTAMPDIFF(SECOND,starttime,NOW()), errormessage=CONCAT('error=',r_code,', message=',r_msg),randno=NULL WHERE starttime=v_starttime AND randno=v_randno;
	
END$$
DELIMITER ;`

通过查询mysql.t_event_history表，我们就知道event何时执行，执行是否成功，执行时长，出错时的错误信息，为管理我们日常调度计划提供很大方便。

转载自https://blog.csdn.net/seteor/article/details/19411717#