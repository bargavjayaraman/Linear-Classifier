#include<iostream.h>
#include<fstream.h>
#include<conio.h>
#include"ClassifierStructure.h"
#include<process.h>

int main()
{
    DataSet ds1,ds2,ds3;
    float error,std_dev=0;
    int ch=0,a,b;
    while(ch!=-1)
    {
                 cout<<"\nEnter your choice:\n";
                 cout<<"1. Iris DataSet\n2. Balance Scale DataSet\n3. Haberman DataSet\n4. Exit\n";
                 cin>>ch;
                 a=b=0;
                 switch(ch)
                 {
                           case 1:
                                            ds1.readData("iris-data.txt",150,4);
                                            cout<<"\nEnter the Algo:\n1. Single Perceptron\n2. Batch Perceptron\n3. Single Relaxation\n";
                                            cout<<"4. Batch Relaxation\n5. MSE using Pseudo Inverse\n6. MSE using LMS\n";
                                            while(a<1 || a>6)
                                               cin>>a;
                                            cout<<"Enter the Combi:\n1. 1 vs Rest\n2. 1 vs 1 Majority Voting\n3. 1 vs 1 DDAG\n4. BHDT\n";
                                            while(b<1 || b>4)
                                               cin>>b;
                                            cout<<"\nIris Data Set:\n";
                                            error=crossValidate(ds1,150,4,3,5,a,b,std_dev);
                                            cout<<"\nAverage Error:\t"<<error<<"\nStandard Deviation:\t"<<std_dev<<endl;
                                            break;
                           case 2:
                                            ds2.readData("balance-scale-data.txt",625,4);
                                            cout<<"\nEnter the Algo:\n1. Single Perceptron\n2. Batch Perceptron\n3. Single Relaxation\n";
                                            cout<<"4. Batch Relaxation\n5. MSE using Pseudo Inverse\n6. MSE using LMS\n";
                                            while(a<1 || a>6)
                                               cin>>a;
                                            cout<<"Enter the Combi:\n1. 1 vs Rest\n2. 1 vs 1 Majority Voting\n3. 1 vs 1 DDAG\n4. BHDT\n";
                                            while(b<1 || b>4)
                                               cin>>b;
                                            cout<<"\nBalance Scale Data Set:\n";
                                            error=crossValidate(ds2,625,4,3,5,a,b,std_dev);
                                            cout<<"\nAverage Error:\t"<<error<<"\nStandard Deviation:\t"<<std_dev<<endl;
                                            break;
                           case 3:
                                            ds3.readData("haberman-data.txt",305,3);
                                            cout<<"\nEnter the Algo:\n1. Single Perceptron\n2. Batch Perceptron\n3. Single Relaxation\n";
                                            cout<<"4. Batch Relaxation\n5. MSE using Pseudo Inverse\n6. MSE using LMS\n";
                                            while(a<1 || a>6)
                                               cin>>a;
                                            cout<<"Enter the Combi:\n1. 1 vs Rest\n2. 1 vs 1 Majority Voting\n3. 1 vs 1 DDAG\n4. BHDT\n";
                                            while(b<1 || b>4)
                                               cin>>b;
                                            cout<<"\nHaberman Data Set:\n";
                                            error=crossValidate(ds3,305,3,2,5,a,b,std_dev);
                                            cout<<"\nAverage Error:\t"<<error<<"\nStandard Deviation:\t"<<std_dev<<endl;
                                            break;
                           default:
                                            exit(0);
                 }
    }
    getch();
    return 0;
}
