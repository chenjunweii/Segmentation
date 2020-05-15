#!/bin/bash
cp ./prediction_csc15.txt ./dataset/NCU_NLPLab_CSC/sighan8csc_release1.0/Test/
cd ./dataset/NCU_NLPLab_CSC/sighan8csc_release1.0/Tool

echo $(ls)
echo $(pwd)

#echo $(java -jar sighan15csc.jar)
java -jar sighan15csc.jar -t ../Test/SIGHAN15_CSC_TestTruth.txt -i ../Test/prediction_csc15.txt
