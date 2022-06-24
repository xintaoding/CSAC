# Core Sample Consensus Method for Two-view Correspondence Matching

Anhui Normal University
School of Computer and Information, Wuhu, Anhui, China
241002
Xintao Ding (xintaoding@163.com)


## Usage

Requirement:

	1.cuda 8.0

	2.visual studio 2015

Run:
Homography, essential matrix estimation is run by one example.

Fundamental matrix estimation is run on kusvod2 dataset by matlab_launch_cexe.m
If you need to run an example, please uncomment line 1573:
//char filename[] = "booksh.txt";
and comment the statement that receives filename from main function at line 1574-1575: 
	char *filename;//designed for receiving string from main
	filename = argv[1];//designed for receiving string from main

In fundamental matrix estimation, recall and precision is calculated in recall_precision.m 

The registration results are shown by result_show_kusvod2_csac.m
