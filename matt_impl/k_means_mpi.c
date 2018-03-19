/* File:     k_means_mpi.c
 *
 * Purpose:  A parallel program for 1-d k-means
 *
 * Compile:  mpicc k_means_mpi k_means_mpi.c
 * Usage:    mpiexec -n <number of processes> ./k_means_mpi
 *
 * Input:    Data, k clusters
 * Output:   k centroids
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct cent_weight{
	double centroid;
	int weight;
};

int Compare(const void* a_p, const void* b_p);

/*-------------------------------------------------------------------*/

int main(void) {
   
   int p,id;
   int k = 2;
   
   double data[] = {3,12,25,30,41,48,53,58,94,99,
					1,13,20,23,26,32,39,54,74,75,
					18,27,37,41,47,52,58,70,75,94};

   MPI_Init(NULL, NULL);
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   MPI_Comm_rank(MPI_COMM_WORLD, &id);
   
   int n = sizeof(data) / sizeof(data[0]);
   int part = n/p;
   int extra = n%p;
   int local_n = part;
   if (id < extra)
	   local_n += 1;
   double* local_data;
   double* centroids;
   double* all_centroids;
   int* assign;
   int* weights;
   int* all_weights;
   struct cent_weight *all_centw = malloc(k*p*sizeof(struct cent_weight));
   int i, j, change;
   int d,c,div;
   double dist, cur_dist, sum;
   int sendcounts[p];
   int displs[p];
   int disp_sum = 0;
   
   local_data = malloc(local_n*sizeof(double));
   centroids = malloc(k*sizeof(double));
   all_centroids = malloc(k*p*sizeof(double));
   assign = malloc(local_n*sizeof(int));
   weights = malloc(k*sizeof(int));
   all_weights = malloc(k*p*sizeof(int));
   
   for (i=0;i<p;i++){
	   sendcounts[i] = part;
	   if (i<extra)
		   sendcounts[i]++;
	   displs[i] = disp_sum;
	   disp_sum += sendcounts[i];
   }
   
   MPI_Scatterv(data, sendcounts, displs, MPI_DOUBLE, 
			local_data, sendcounts[id], MPI_DOUBLE, 0, MPI_COMM_WORLD);

   for (i = 0; i < k; i++){
	   centroids[i] = local_data[i];
	   weights[i] = 0;
   }
   
   for (i = 0; i < local_n; i++){
	   assign[i] = 0;
   }
   change = 1;
   
   while (change != 0){
	   change = 0;
	   for (d = 0; d < local_n; d++){
		   dist = abs(local_data[d]-centroids[assign[d]]);
		   for (c = 0; c<k; c++){
			   if (assign[d] != c){
				   cur_dist = abs(local_data[d]-centroids[c]);
				   if (cur_dist < dist){
					   dist = cur_dist;
					   assign[d] = c;
					   change = 1;
				   }
			   }
		   }
	   }
	   
	   for (c = 0; c < k; c++){
		   sum = 0;
		   div = 0;
		   for (d = 0; d < local_n; d++){
			   if (assign[d] == c){
				   sum += local_data[d];
				   div += 1;
			   }
		   }
		   centroids[c] = sum/(double)div;
	   }
	   
   }  /*while*/
   
   for (i=0; i< local_n; i++)
	    weights[assign[i]] += 1; 
	
   MPI_Gather(centroids, k, MPI_DOUBLE, 
			all_centroids,k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
   MPI_Gather(weights, k, MPI_INT, 
			all_weights,k, MPI_INT, 0, MPI_COMM_WORLD);
   
   if (id == 0){
	   
	   for (i=0;i<p*k;i++){
		   all_centw[i].centroid = all_centroids[i];
		   all_centw[i].weight = all_weights[i];
	   }
	   
	   qsort(all_centw,k*p,sizeof(struct cent_weight), Compare);			  
				  
	   for (c=0;c<k;c++){
		   sum = 0;
		   div = 0;
		   for (i=c*p;i<(c+1)*p;i++){
			   sum += all_centw[i].centroid*all_centw[i].weight;
			   div += all_centw[i].weight;
		   }
		   centroids[c] = sum/(double)div;
		   printf("Final Centroid %d = %f \n",c,centroids[c]);
	   }
   }
   
   free(local_data);
   free(centroids);
   free(all_centroids);
   free(assign);
   free(weights);

   MPI_Finalize();
   
   return 0;

}  /* main */

/*-------------------------------------------------------------------
 * Function:    Compare
 * Purpose:     Compare 2 doubles, return -1, 0, or 1, respectively, when
 *              the first double is less than, equal, or greater than
 *              the second.  Used by qsort.
 */
int Compare(const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}