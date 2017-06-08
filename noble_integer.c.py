/**
 * @input A : Integer array
 * @input n1 : Integer array's ( A ) length
 * 
 * @Output Integer
 */
 
void insertionSort(int arr[],int n)
{
    int i,j,key;
    
    
    for(i=1;i<n;i++)
    {
        key = arr[i];
        j = i-1;
        while(arr[j] < key && j >=0)
        {
            arr[j+1] = arr[j];
            j--;
            
        }
        arr[j+1] = key;
    }
        
    
}
int solve(int* A, int n1) {
   
   
   
   insertionSort(A,n1);
   
   int i,flag;
   flag = 0 ;
   
   for(i=0;i<n1;i++)
   {
       if(i == A[i])
            flag = 1;
   }
   
   if(flag ==0 )
   {
       return -1;
   }
   else
   {
       return 1;
   }
   
   // apply insertion sort 
   
   
   
   
}
