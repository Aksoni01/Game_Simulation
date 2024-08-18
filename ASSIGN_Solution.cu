
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here
__global__ void dFiring(int *gpucoordX, int *gpucoordY, int *gpuScore, int *gpuHealth, int T, int M, int *health,int *tank_alive,long int round) {
   
    *tank_alive=T;

    if(round % T ==0 || health[blockIdx.x] <= 0 ) return;

    __shared__ long long int previous_distance;
    previous_distance=LLONG_MAX;
    __syncthreads();
    int source = blockIdx.x ,destination = (blockIdx.x + round) %T;

    long long int global_dest_id = (long long int)gpucoordY[destination] * M + gpucoordX[destination];
    long long int global_source_id =(long long int) gpucoordY[source] * M + gpucoordX[source];
    long long int global_temp_id =(long long int)  gpucoordY[threadIdx.x] * M + gpucoordX[threadIdx.x];
    long long int curr= abs(global_temp_id - global_source_id );

    long long int val = (long long int) gpucoordX[source] * (gpucoordY[destination] - gpucoordY[threadIdx.x]) +
                            (long long int)  gpucoordX[destination] * (gpucoordY[threadIdx.x] - gpucoordY[source]) +
                            (long long int)  gpucoordX[threadIdx.x] * (gpucoordY[source] - gpucoordY[destination]);

    if( (health[threadIdx.x]>0 && global_dest_id > global_source_id && global_temp_id > global_source_id &&  val == 0 ) || (health[threadIdx.x]>0 && global_dest_id < global_source_id && global_temp_id < global_source_id && val == 0 ) ){
                atomicMin(&previous_distance, curr );
    }
    __syncthreads();

    if(  (global_dest_id > global_source_id && global_temp_id > global_source_id && previous_distance == curr ) || (global_dest_id < global_source_id && global_temp_id < global_source_id &&  previous_distance == curr) ){
        atomicSub(&gpuHealth[threadIdx.x],1);
        atomicAdd(&gpuScore[source],1);
    }
}

__global__ void dCount(int * gpuHealth,int *tank_alive){
  if(gpuHealth[threadIdx.x]<=0){
    atomicSub(tank_alive,1);
  }
}


__global__ void memoryallote(int * gpuHealth,int * gpuScore,int H){
  gpuHealth[threadIdx.x]=H;
  gpuScore[threadIdx.x]=0;
}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
  
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int *gpucoordX,*gpucoordY,*gpuScore,*gpuHealth;
    cudaMalloc(&gpucoordX,T * sizeof(int));
    cudaMalloc(&gpucoordY,T * sizeof(int));
    cudaMalloc(&gpuScore,T * sizeof(int));
    cudaMalloc(&gpuHealth,T * sizeof(int));
    cudaMemcpy(gpucoordX,xcoord,T * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpucoordY,ycoord,T * sizeof(int),cudaMemcpyHostToDevice);

    
    memoryallote<<<1,T>>>(gpuHealth,gpuScore,H);
    cudaDeviceSynchronize();

    int *tank_alive;
    long int round=1;
    cudaHostAlloc(&tank_alive, sizeof(int), 0); 
    int *health;
    cudaMalloc(&health,T * sizeof(int));

     do{
      cudaMemcpy(health, gpuHealth, T * sizeof(int), cudaMemcpyDeviceToDevice);
      dFiring<<<T, T>>>(gpucoordX, gpucoordY, gpuScore, gpuHealth, T, M+1, health,tank_alive,round);
      dCount<<<1, T>>>(gpuHealth, tank_alive);
      cudaDeviceSynchronize();
      round++;
    }while (*tank_alive >= 2 );

    cudaMemcpy(score,gpuScore,T * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(health);
    cudaFree(gpuHealth);
    cudaFree(gpucoordX);
    cudaFree(gpucoordY);
    cudaFree(gpuScore);
    cudaFreeHost(tank_alive);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
