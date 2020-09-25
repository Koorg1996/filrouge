#CREATE DATABASE
CREATE KEYSPACE MYBASE IF NOT EXISTS WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor': 1};
USE MYBASE;

#CREATE TABLES
CREATE TABLE MYTABLE1 (<rows>, primary key (<prows>));
COPY MYTABLE1 (<csv_rows>) FROM <csv_emplacement> WITH DELIMITER = ',' AND HEADER = TRUE;
CREATE TABLE MYTABLE2 (<rows>, primary key (<prows>));
COPY MYTABLE2 (<csv_rows>) FROM <csv_emplacement> WITH DELIMITER = ',' AND HEADER = TRUE;
#...
<work>

#EXPORT RESULT
COPY MYTABLE1 TO '/csvfiles/result1.csv';
COPY MYTABLE2 TO '/csvfiles/result2.csv';
#...

