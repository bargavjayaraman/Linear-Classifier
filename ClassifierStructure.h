#include"newmatap.h"
#include"newmatio.h"

class DataItem{
      public:
	int classLabel;
	float *feature;
};


// Container for data feature vectors, class labels for each
// feature vector and functions to read and write data.
// Use constructor and destructor to initialize and clear data.
class DataSet{
   public:
  // Assume that the data file is plain text with each row
  // containing the class label followed by the features, 
  // separated by blank spaces.
  bool readData(char *filename,int entry_num,int attr_num)
  {
    fstream ob;
    data=new DataItem[entry_num];
    for(int i=0;i<entry_num;i++)
       data[i].feature=new float[attr_num+1];
    ob.open(filename, ios::in);
    if(!ob)
       return false;
    while(ob)
    {
       for(int k=0;k<entry_num;k++)
       {
               ob>>data[k].classLabel;
               for(int i=0;i<attr_num;i++)
                  ob>>data[k].feature[i];
               data[k].feature[attr_num]=1.0;
       }
    }
    ob.close();
    return true;
  }

  // Write data in the above format.
  //bool writeData(filename);

  // Variables for data
  DataItem* data;
  
};


// Partition 'complete' dataset randomly into 'folds' parts and 
// returns a pointer to an array of pointers to the partial datasets.
// seed is an argument to random number generator. The function can
// be used to divide data for training, testing and cross validation.
// This need not replicate the data.
DataSet* splitDataset(int entry_num, DataSet complete, int folds=2, int seed=0)
{
     DataSet *splitSet;
     int *ra,x,counter=0;
     ra=new int[entry_num];
     splitSet=new DataSet[folds];
     for(int i=0;i<folds;i++)
        splitSet[i].data=new DataItem[entry_num/folds];
     for(int i=0;i<entry_num;i++)
      ra[i]=-1;
     srand(seed);
     do
     {
       x=rand()%entry_num;
       for(int i=0;i<entry_num;i++)
       {
         if(ra[i]==-1)
         {
           ra[i]=x;
           counter++;
           break;
         }
         else if(ra[i]==x)
           break;
       }
     }while(counter<entry_num);
     for(int i=0;i<5;i++)
       for(int j=0;j<entry_num/folds;j++)
           splitSet[i].data[j]=complete.data[ra[j+i*(entry_num/folds)]];
     return splitSet;
}

// Merge the datasets indexed by indicesToMerge in the toMerge list and return a
// single dataset. This need not replicate the data.
DataSet mergeDatasets(DataSet* toMerge, int numDatasets, int entry_num, int* indicesToMerge)
{
         DataSet merge;
         merge.data=new DataItem[numDatasets*entry_num];
         for(int i=0;i<numDatasets;i++)
            for(int j=0;j<entry_num;j++)
               merge.data[i*entry_num+j]=toMerge[indicesToMerge[i]].data[j];
         return merge;
}
// Class that carries out training and classification as well as
// store and read the learned model in/from a file.
class LinearClassifier{
public:

  // Loads classifier model from a file
  bool loadModel(int class_num, int attr_num, char *modelfilename)
  {
       ifstream fin;
       //fin=new ifstream;
       fin.open(modelfilename, ios::in);
       for(int i=0;i<class_num;i++)
               for(int j=0;j<attr_num;j++)
                       fin>>classifier[i][j];
       return 1;
  }

  // Saves the learned model parameters into a file
  bool saveModel(int class_num, int attr_num, char *modelfilename)
  {
       ofstream fout;
       //fout=new ofstream;
       fout.open(modelfilename, ios::out);
       for(int i=0;i<class_num;i++)
       {
               for(int j=0;j<attr_num;j++)
                       fout<<classifier[i][j]<<' ';
               fout<<'\n';
       }
       return 1;
  }

  // learn the parameters of the classifier from possibly multiple training datasets
  // using a specific learning algorithm and combination strategy. 
  // The function should return the training error in [0,1].
  // Algorithms:
  //	1: Single Sample Perceptron Learning (fixed eta)
  //	2: Batch Perceptron Learning (variable eta)
  //	3: Single sample Relaxation (variable eta)
  //	4: Batch Relaxation Learning (variable eta)
  //	5: MSE using Pseudo Inverse
  //	6: MSE using LMS Procedure
  // Combination:
  //	1: 1 vs. Rest
  //	2: 1 vs. 1 with Majority voting
  //	3: 1 vs. 1 with DDAG
  //	4: BHDT.2
  float learnModel(DataSet trainSet,int entry_num, int attr_num, int class_num, int algorithm, int combination)
  {
        int right;
        float sum;
        for(int k=0;k<class_num;k++)
           for(int i=0;i<attr_num+1;i++)
              classifier[k][i]=1.0;
        combination=(combination==4)?1:combination;
        combination=(combination==3)?2:combination;
        if(algorithm==1)
        {
           for(int k=1;k<=class_num;k++)
           {
              right=0;
              for(int i=0;i<entry_num;i++)
              {
                 if(combination==1 && trainSet.data[i].classLabel!=k)
                    for(int j=0;j<attr_num+1;j++)
                       trainSet.data[i].feature[j]*=-1;
                 else if(combination==2)
                 {
                      if(class_num==3 && trainSet.data[i].classLabel==((k+1)%class_num)+1)
                         continue;
                      if(trainSet.data[i].classLabel==(k%class_num)+1)
                         for(int j=0;j<attr_num+1;j++)
                            trainSet.data[i].feature[j]*=-1;
                 }
                 sum=0;
                 for(int j=0;j<attr_num+1;j++)
                    sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                 if(sum<0)
                    for(int j=0;j<attr_num+1;j++)
                       classifier[k-1][j]+=trainSet.data[i].feature[j];
                 else
                    right++;
                 if(combination==1 && trainSet.data[i].classLabel!=k)
                    for(int j=0;j<attr_num+1;j++)
                       trainSet.data[i].feature[j]*=-1;
                 else if(combination==2 && trainSet.data[i].classLabel==(k%class_num)+1)
                    for(int j=0;j<attr_num+1;j++)
                       trainSet.data[i].feature[j]*=-1;
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
        else if(algorithm==2)
        {
           float *temp_arr,eta;
           int counter;
           temp_arr=new float[attr_num+1];
           for(int k=1;k<=class_num;k++)
           {
              counter=0;
              eta=1;
              while(eta>0.001)
              {
                 counter++;
                 right=0;
                 for(int i=0;i<attr_num+1;i++)
                    temp_arr[i]=0;
                 for(int i=0;i<entry_num;i++)
                 {
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2)
                    {
                       if(class_num==3 && trainSet.data[i].classLabel==((k+1)%class_num)+1)
                          continue;
                       if(trainSet.data[i].classLabel==(k%class_num)+1)
                          for(int j=0;j<attr_num+1;j++)
                            trainSet.data[i].feature[j]*=-1;
                    }
                    sum=0;
                    for(int j=0;j<attr_num+1;j++)
                       sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                    if(sum<0)
                       for(int j=0;j<attr_num+1;j++)
                          temp_arr[j]+=trainSet.data[i].feature[j];
                    else
                       right++;
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2 && trainSet.data[i].classLabel==(k%class_num)+1)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                 }
                 eta=(float)1/counter;
                 for(int i=0;i<attr_num+1;i++)
                    classifier[k-1][i]+=eta*temp_arr[i];
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
        else if(algorithm==3)
        {
           int counter;
           float eta;
           for(int k=1;k<=class_num;k++)
           {
              counter=0;
              eta=1.0;
              while(eta>0.001)
              {
                 right=0;
                 for(int i=0;i<entry_num;i++)
                 {
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2)
                    {
                       if(class_num==3 && trainSet.data[i].classLabel==((k+1)%class_num)+1)
                          continue;
                       if(trainSet.data[i].classLabel==(k%class_num)+1)
                          for(int j=0;j<attr_num+1;j++)
                             trainSet.data[i].feature[j]*=-1;
                    }
                    sum=0;
                    for(int j=0;j<attr_num+1;j++)
                       sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                    if(sum<=1.0)
                    {
                       counter++;
                       eta=(float)1/counter;
                       float sum1=0.0;
                       for(int j=0;j<attr_num+1;j++)
                          sum1+=trainSet.data[i].feature[j]*trainSet.data[i].feature[j];
                       for(int j=0;j<attr_num+1;j++)
                          classifier[k-1][j]+=eta*(1.0-sum)*trainSet.data[i].feature[j]/sum1;
                    }
                    else
                       right++;
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2 && trainSet.data[i].classLabel==(k%class_num)+1)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                 }
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
        else if(algorithm==4)
        {
           float *temp_arr,eta;
           int counter;
           temp_arr=new float[attr_num+1];
           for(int k=1;k<=class_num;k++)
           {
              counter=0;
              eta=1.0;
              while(eta>0.001)
              {
                 right=0;
                 for(int i=0;i<attr_num+1;i++)
                    temp_arr[i]=0;
                 for(int i=0;i<entry_num;i++)
                 {
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2)
                    {
                       if(class_num==3 && trainSet.data[i].classLabel==((k+1)%class_num)+1)
                          continue;
                       if(trainSet.data[i].classLabel==(k%class_num)+1)
                          for(int j=0;j<attr_num+1;j++)
                             trainSet.data[i].feature[j]*=-1;
                    }
                    sum=0;
                    for(int j=0;j<attr_num+1;j++)
                       sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                    if(sum<=1.0)
                    {
                       counter++;
                       eta=(float)1/counter;
                       float sum1=0.0;
                       for(int j=0;j<attr_num+1;j++)
                          sum1+=trainSet.data[i].feature[j]*trainSet.data[i].feature[j];
                       for(int j=0;j<attr_num+1;j++)
                          temp_arr[j]+=eta*(1.0-sum)*trainSet.data[i].feature[j]/sum1;
                    }  
                    else
                       right++;
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2 && trainSet.data[i].classLabel==(k%class_num)+1)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                 }
                 for(int i=0;i<attr_num+1;i++)
                    classifier[k-1][i]+=temp_arr[i];
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
        else if(algorithm==5)
        {
           Real array[entry_num*(attr_num+1)];
           Real b_arr[entry_num];
           Matrix F(entry_num,attr_num+1);
           Matrix temp(attr_num+1,attr_num+1);
           Matrix Res(attr_num+1,entry_num);
           Matrix b(entry_num,1);
           Matrix a(attr_num+1,1);
           IdentityMatrix I(attr_num+1);
           for(int i=0;i<entry_num;i++)
              b_arr[i]=1.0;
           b<<b_arr;
           for(int k=1;k<=class_num;k++)
           {
              right=0;
              for(int i=0;i<entry_num;i++)
              {
                 if(trainSet.data[i].classLabel!=k)
                    for(int j=0;j<attr_num+1;j++)
                       array[i*(attr_num+1)+j]=(Real)(-1*trainSet.data[i].feature[j]);
                 else
                    for(int j=0;j<attr_num+1;j++)
                       array[i*(attr_num+1)+j]=(Real)trainSet.data[i].feature[j];
              }
              F<<array;
              temp=F.t()*F;
              if(temp.Determinant()==0)
              {
                 temp+=0.1*I;
                 cout<<"\nDeterminant 0!!";
                 getch();
                 //exit(0);
              }
              Res=temp.i()*F.t();
              a=Res*b;
              for(int j=0;j<attr_num+1;j++)
                 classifier[k-1][j]=(float)a(j+1,1);
              for(int i=0;i<entry_num;i++)
              {
                 sum=0.0;
                 for(int j=0;j<attr_num+1;j++)
                    sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                 if(sum>1.0)
                    right++;
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
        else if(algorithm==6)
        {
           int counter;
           float eta,alpha;
           for(int k=1;k<=class_num;k++)
           {
              counter=0;
              eta=1.0;
              alpha=1.0;
              while(alpha>0.001 && eta>0.001)
              {
                 right=0;
                 for(int i=0;i<entry_num;i++)
                 {
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2)
                    {
                       if(class_num==3 && trainSet.data[i].classLabel==((k+1)%class_num)+1)
                          continue;
                       if(trainSet.data[i].classLabel==(k%class_num)+1)
                          for(int j=0;j<attr_num+1;j++)
                             trainSet.data[i].feature[j]*=-1;
                    }
                    sum=0.0;
                    for(int j=0;j<attr_num+1;j++)
                       sum+=classifier[k-1][j]*trainSet.data[i].feature[j];
                    counter++;
                    eta=(float)1/counter;
                    if(sum<=1.0)
                    {
                       alpha=0.0;
                       for(int j=0;j<attr_num+1;j++)
                       {
                          alpha+=eta*(1.0-sum)*trainSet.data[i].feature[j];
                          classifier[k-1][j]+=eta*(1.0-sum)*trainSet.data[i].feature[j];
                       }
                    }
                    else
                       right++;
                    if(combination==1 && trainSet.data[i].classLabel!=k)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                    else if(combination==2 && trainSet.data[i].classLabel==(k%class_num)+1)
                       for(int j=0;j<attr_num+1;j++)
                          trainSet.data[i].feature[j]*=-1;
                 }
              }
           }
           saveModel(class_num, attr_num+1, "modelfile.txt");
           return (float)(entry_num-right)/entry_num;
        }
  }
  
  // Classifies a DataItem and returns the class-label
  int classifySample(DataItem data, int attr_num, int class_num, int combi)
  {
      int a[class_num],x,y;
      loadModel(class_num, attr_num, "modelfile.txt");
      if(combi==4)
      {
         float sum=0.0;
         if(class_num==2)
         {
            for(int i=0;i<attr_num;i++)
               sum+=classifier[0][i]*data.feature[i];
            if(sum>0)
               return 1;
            else
               return 2;
         }
         else
         {
            for(int i=0;i<attr_num;i++)
               sum+=classifier[0][i]*data.feature[i];
            if(sum>0)
               return 1;
            else
            {
                sum=0.0;
                for(int i=0;i<attr_num;i++)
                   sum+=classifier[1][i]*data.feature[i];
                if(sum>0)
                   return 2;
                else
                   return 3;
            }
         }
      }
      if(combi==3)
      {
         float sum=0.0;
         if(class_num==2)
         {
            for(int i=0;i<attr_num;i++)
               sum+=classifier[0][i]*data.feature[i];
            if(sum>0)
               return 1;
            else
               return 2;
         }
         else
         {
            for(int i=0;i<attr_num;i++)
               sum+=classifier[2][i]*data.feature[i];
            if(sum>0)
            {
               sum=0.0;
               for(int i=0;i<attr_num;i++)
                  sum+=classifier[1][i]*data.feature[i];
               if(sum>0)
                  return 2;
               else
                  return 3;
            }
            else
            {
               sum=0.0;
               for(int i=0;i<attr_num;i++)
                  sum+=classifier[0][i]*data.feature[i];
               if(sum>0)
                  return 1;
               else
                  return 2;
            }
         }
      }
      for(int i=0;i<class_num;i++)
         a[i]=0;
      for(int k=0;k<class_num;k++)
      {
              float sum=0.0;
              for(int i=0;i<attr_num;i++)
                 sum+=classifier[k][i]*data.feature[i];
              if(sum>0)
                 a[k]+=1;
              else
              {
                  if(combi==2)
                     a[(k+1)%class_num]+=1;
                  else
                  {
                     for(int i=0;i<class_num;i++)
                        if(i!=k)
                           a[i]+=1;
                  }
              }
      }
      x=0;y=a[0];
      for(int i=1;i<class_num;i++)
         if(y<a[i])
            x=i;
      return x+1;
  }
  // classify a set of testDataItems and return the error rate in [0,1].
  // Also fill the entries of the confusionmatrix.
  float classifyDataset(DataSet testSet,int entry_num, int attr_num, int class_num, int combi)
  {
        int cm[3][3];
        for(int i=0;i<class_num;i++)
                for(int j=0;j<class_num;j++)
                        cm[i][j]=0;
        int class_type,right=0;
        for(int i=0;i<entry_num;i++)
        {
           class_type=classifySample(testSet.data[i], attr_num+1, class_num, combi);
           if(class_type==testSet.data[i].classLabel)
             right++;
           cm[testSet.data[i].classLabel-1][class_type-1]+=1;
        }
        cout<<"\nConfusion Matrix:\n";
        for(int i=0;i<class_num;i++)
        {
                for(int j=0;j<class_num;j++)
                        cout<<cm[i][j]<<'\t';
                cout<<endl;
        }
        return (float)(entry_num-right)/entry_num;     
  }
  
  // Variablesclassify
  // Other variables to hold classifier parameters.
  float classifier[3][5];

};

// Divide the dataset and performa an n-fold cross-validation. Compute the
// average error rate in [0,1]. Fill in the standard deviation and confusion matrix.
float crossValidate(DataSet complete, int entry_num, int attr_num, int class_num, int folds, 
			int algo, int comb, float &std_dev)
{
    DataSet *split, merge;
    int indices[folds-1];
    float error[2*folds],tot_error=0.0;
    LinearClassifier l;
    split=splitDataset(entry_num,complete,folds);
    for(int i=0;i<folds;i++)
    {
            for(int j=0,k=0;j<folds;j++)
            {
                    if(j!=i)
                            indices[k++]=j;
            }
            cout<<"\nIteration "<<i+1<<":\n\n";
            merge=mergeDatasets(split,folds-1,entry_num/folds,indices);
            error[i]=l.learnModel(merge,(folds-1)*entry_num/folds,attr_num,class_num,algo,comb);
            tot_error+=error[i];
            cout<<"Training Error:\t"<<error[i];
            error[i+folds]=l.classifyDataset(split[i],entry_num/folds,attr_num,class_num,comb);
            tot_error+=error[i+folds];
            cout<<"Testing Error:\t"<<error[i+folds]<<endl;
    }

    std_dev=0.0;
    for(int i=0;i<2*folds;i++)
            std_dev+=(tot_error/(2*folds)-error[i])*(tot_error/(2*folds)-error[i]);
    std_dev/=(2*folds);
    return tot_error/(2*folds);
}


