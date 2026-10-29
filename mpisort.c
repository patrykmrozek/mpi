#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include "mpi.h"

double * merge_array(int n, double * a, int m, double * b);
void     merge_sort(int n, double * a);
void     swap (double * a, double * b);
void     bubble_sort(int n, double * a);

int MPI_Sort_direct(int n, double * a, int root, MPI_Comm comm);
int MPI_Sort_bucket(int n, double * a, double max, int root, MPI_Comm comm);
int MPI_Sort_odd_even(int n, double * array, int root, MPI_Comm comm);
int MPI_Sort_shell(int n, double * array, int root, MPI_Comm comm);
int MPI_Sort_bitonic(int n, double * array, int root, MPI_Comm comm, double *tcomm_out, double *tcomp_out);

int MPI_Exchange(int n, double * array, int rank1, int rank2, MPI_Comm comm, double *tcomm_out, double *tcomp_out);
int MPI_Is_sorted(int n, double *array, int *result, int root, MPI_Comm comm);

static inline void
logv(const char *str, ...)
{
    va_list args;
    va_start(args, str);
    vfprintf(stdout, str, args); 
    fputc('\n', stdout);
    va_end(args);
}

#define LOG(s, ...) \
    do { \
        logv(s, ##__VA_ARGS__); \
    } while(0)

#ifdef DEBUG_LOG

#define DEBUG(s, ...) \
    do { \
        logv("%s:%d "s, \
             __FUNCTION__, __LINE__, \
             ##__VA_ARGS__); \
    } while(0)

#define DEBUG_RANK(r, s, ...) \
    do { \
        logv("PROC:%d %s:%d "s, \
             (r), \
             __FUNCTION__, __LINE__, \
             ##__VA_ARGS__); \
    } while(0)

#else

#define DEBUG(s, ...)         ((void)0)
#define DEBUG_RANK(r, s, ...) ((void)0)

#endif //DEBUG_LOG

int main(int argc, char *argv[])
{
	int rank, size;
    int ret = MPI_SUCCESS;
	int n = 10000, i, j, k, x, q, l, shell, pair, *nr;
	double m = 10.0;
	double * scattered_array, * array;

	// Init + rank + size
	MPI_Init(&argc, &argv);
   	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   	MPI_Comm_size(MPI_COMM_WORLD, &size);

    array = (double *) calloc(n, sizeof(double));
	if(rank == 0)
	{

	   //initialise the array with random values, then scatter to all processors
	   srand(((unsigned)time(NULL)+rank));

	   for(i = 0; i < n; i++)
	   {
	      array[i]=((double)rand()/RAND_MAX)*m;
	   }

	}

    // call and time evaluate MPI_Sort_direct
    double tcomm, tcomp;
    // MPI_Sort_odd_even(n, array, 0, MPI_COMM_WORLD);
    // ret = MPI_Sort_shell(n, array, 0, MPI_COMM_WORLD);
    ret = MPI_Sort_bitonic(n, array, 0, MPI_COMM_WORLD, &tcomm, &tcomp);

    if (ret != MPI_SUCCESS) {
        LOG("MPI failure: %d", ret);
        goto final;
    }
    
    LOG("Proc %d: Comm Time: %lf Comp Time: %lf", rank, tcomm, tcomp);

final:
	MPI_Finalize();
}

// MPI Functions

int MPI_Sort_direct(int n, double * array, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    double * localArray = (double *)calloc(n/size, sizeof(double));
    
    // scatter array to localArray with double elements
    int rc = MPI_Scatter(array, n/size, MPI_DOUBLE, localArray, n/size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS)return rc;

    // sequential sort localArray
    bubble_sort(n/size, localArray);

    // gather localArray to array with double elements
    rc = MPI_Gather(localArray, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    if(rc != MPI_SUCCESS)return rc;

    if(rank == 0) {
        // merge the size chunks of array
        for(int i=1;i<size;i++) {
            double * tmpArray = merge_array(n/size*i, array, n/size, array+n/size*i);
            for(int j=0;j<(i+1)*n/size;j++)array[j] = tmpArray[j];
        }
    }
    
    return MPI_SUCCESS;
}

int MPI_Sort_bucket(int n, double * a, double max, int root, MPI_Comm comm)
{
    int     rank, size;
    double *bucket;
    int     bucket_count;
    int    *recv_counts, *displacements;
    int     ret = MPI_SUCCESS;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    bucket_count  = 0;
    bucket        = (double *)calloc(n, sizeof(*bucket));
    recv_counts   = (int *)calloc(size, sizeof(*recv_counts));
    displacements = (int *)calloc(size, sizeof(*displacements));

    if (!bucket || !recv_counts || !displacements) {
        ret = MPI_T_ERR_MEMORY;
        goto out;
    }

    /* bcast a */
    ret = MPI_Bcast(a, n, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    /* 
     * collect as elems to proc ranks bucket
     * since we don;t know how big the bucket could be we init to n
     */
    for (int i = 0; i < n; i++) {
        /* check if a[i] is in the bucket [max/size*rank, max/size*(rank+1)] */
        if (a[i] >= max/size+rank && a[i] < max/size*(rank+1)) {
            bucket[bucket_count++] = a[i];
        }
    }

    /* sort bucket */
    merge_sort(bucket_count, bucket);

    /* gatherv */
    ret = MPI_Gather(&bucket_count, 1, MPI_INT, recv_counts, 1, MPI_INT, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    if (rank == root) {
        displacements[0] = 0;
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    ret = MPI_Gatherv(bucket, bucket_count, MPI_DOUBLE,
                      a, recv_counts, displacements, MPI_DOUBLE,
                      root, comm);

    if (ret != MPI_SUCCESS) goto out;

out:
    if (bucket) free(bucket);
    return ret;
}


// function to merge the array a with n elements with the array b with m elements
// function returns the nerged array
double * merge_array(int n, double * a, int m, double * b)
{
   int i,j,k;
   double * c = (double *) calloc(n+m, sizeof(double));

   for(i=j=k=0;(i<n)&&(j<m);)
      if(a[i]<=b[j])c[k++]=a[i++];
      else c[k++]=b[j++];

   if(i==n)for(;j<m;)c[k++]=b[j++];
   else for(;i<n;)c[k++]=a[i++];

return c;
}

// function to merge sort the array a with n elements

void merge_sort(int n, double * a)
{
   double * c;
   int i;

   if (n<=1) return;

   if(n==2) {
      if(a[0]>a[1])swap(&a[0],&a[1]);
      return;
   }

   merge_sort(n/2,a);merge_sort(n-n/2,a+n/2);
   c=merge_array(n/2,a,n-n/2,a+n/2);
   for(i=0;i<n;i++)a[i]=c[i];

    return;
}

void bubble_sort(int n, double * a)
{
    for(int i=0;i<n-1;i++) {
        for(int j=0;j<n-1;j++) {
            if(a[j]>a[j+1])swap(a+j, a+j+1);
        }
    }
}

// swap two doubles
void swap (double * a, double * b)
{
   double temp;

   temp=*a;*a=*b;*b=temp;
}


int MPI_Sort_odd_even(int n, double * array, int root, MPI_Comm comm)
{
    int rank, size;
    int ret = MPI_SUCCESS;
    double overall_tcomm, overall_tcomp;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    double *local_arr = (double *)calloc(n/size, sizeof(*local_arr));
    if (!local_arr) {
        ret = MPI_T_ERR_MEMORY;
        goto out;
    }
  
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_arr, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    merge_sort(n/size, local_arr);

    for (int step = 0; step < size; step++) {
        if ((step + rank) % 2 == 0) {
            if (rank + 1 < size)
                MPI_Exchange(n/size, local_arr, rank, rank+1, comm, &tcomm, &tcomp);
        } else {
            if (rank > 0)
                MPI_Exchange(n/size, local_arr, rank-1, rank, comm, &tcomm, &tcomp);
        }
        MPI_Barrier(comm);

        int is_sorted;
        ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
        if (ret != MPI_SUCCESS) goto out;

        if (is_sorted) {
            LOG("number of steps: %d", step);
            break;
        }
    }
    ret = MPI_Gather(local_arr, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;
    
out:
    if (local_arr) free(local_arr);

    return ret;
}
 
int MPI_Sort_shell(int n, double *array, int root, MPI_Comm comm)
{
    int rank, size;
    int ret = MPI_SUCCESS;
    double overall_tcomm, overall_tcomp;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    double *local_arr = (double *)calloc(n/size, sizeof(*local_arr));
    if (!local_arr) {
        ret = MPI_T_ERR_MEMORY;
        goto out;
    }
  
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_arr, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    merge_sort(n/size, local_arr);

    //shell
    int left = 0, right = size-1;
    while (left < right) {
        int partner = (right - rank) + left;
        int mid     = (right + left) / 2;

        if (rank < partner) {
            MPI_Exchange(n/size, local_arr, rank, partner, comm, &tcomm, &tcomp);
        } else if (rank > partner){
            MPI_Exchange(n/size, local_arr, partner, rank, comm, &tcomm, &tcomp);
        }
        
        if (rank <= mid) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    int is_sorted;
    ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    if (is_sorted) {
        LOG("sorted by shell");
        goto gather;
    }

    //oddeven
    for (int step = 0; step < size; step++) {
        if ((step + rank) % 2 == 0) {
            if (rank + 1 < size)
                MPI_Exchange(n/size, local_arr, rank, rank+1, comm, &tcomm, &tcomp);
        } else {
            if (rank > 0)
                MPI_Exchange(n/size, local_arr, rank-1, rank, comm, &tcomm, &tcomp);
        }
        MPI_Barrier(comm);

        ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
        if (ret != MPI_SUCCESS) goto out;

        if (is_sorted) {
            LOG("number of steps: %d", step);
            break;
        }
    }

gather:
    ret = MPI_Gather(local_arr, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;
    
out:
    if (local_arr) free(local_arr);

    return ret;
}
 
static inline int
get_dim(int x)
{
    int dim = 0;
    int y = x;
    while (y) {
        dim++;
        y >>= 1;
    }

    return dim;
}

#define BIT_AT(x, o) (((x) >> (o)) & 1)

#define DIR_ASC  0
#define DIR_DESC 1

int MPI_Sort_bitonic(int n,
                     double  *array,
                     int      root,
                     MPI_Comm comm,
                     double  *tcomm_out,
                     double  *tcomp_out)
{
    int    rank, size;
    int    ret = MPI_SUCCESS;
    double overall_tcomm, overall_tcomp;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time = 0.0f;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double *local_arr = (double *)calloc(n/size, sizeof(*local_arr));
    if (!local_arr) {
        ret = MPI_T_ERR_MEMORY;
        goto out;
    }
  
    time = MPI_Wtime();
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_arr, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;
    tcomm += (MPI_Wtime() - time);

    merge_sort(n/size, local_arr);

    time = MPI_Wtime();
    int dim = get_dim(size-1);
    for (int level = 1; level <= dim; level++) {
        int dir = DIR_ASC;
        int group_id = rank >> level;
        int bit = BIT_AT(rank, level);
        for (int i = level; i > 0; i--) {
            dir = bit;
            int pair = rank^(1<<(i - 1));

            DEBUG_RANK(rank, "BIT: %d - ", bit);
            DEBUG_RANK(rank, "RANK: %d, PAIR: %d, LEVEL: %d, GROUPID: %d, DIRECTION: %s",
                    rank, pair, level, group_id, (dir == DIR_ASC ? "ASC" : "DESC"));

            if (dir == DIR_ASC) {
                DEBUG_RANK(rank, "ENTERING exchange ASC");
                if (rank < pair) ret = MPI_Exchange(n/size, local_arr, rank, pair, comm, &tcomm, &tcomp);
                if (rank > pair) ret = MPI_Exchange(n/size, local_arr, pair, rank, comm, &tcomm, &tcomp);
                DEBUG_RANK(rank, "exchanged ASC!");
                if (ret != MPI_SUCCESS) goto out;
            } else {
                DEBUG_RANK(rank, "ENTERING exchange DESC");
                if (rank < pair) ret = MPI_Exchange(n/size, local_arr, pair, rank, comm, &tcomm, &tcomp);
                if (rank > pair) ret = MPI_Exchange(n/size, local_arr, rank, pair, comm, &tcomm, &tcomp);
                DEBUG_RANK(rank, "exchanged DESC!");
                if (ret != MPI_SUCCESS) goto out;
            }
        }

    }
    tcomp += (MPI_Wtime() - time);

    time = MPI_Wtime();
    ret = MPI_Gather(local_arr, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    tcomm += MPI_Wtime() - time;
    if (ret != MPI_SUCCESS) goto out;

    DEBUG_RANK(rank, "gathered!");

    int is_sorted;
    time = MPI_Wtime();
    ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
    tcomm += (MPI_Wtime() - time);
    if (ret != MPI_SUCCESS) goto out;

    DEBUG_RANK(rank, "sorted!");

    ret = MPI_Reduce(&tcomm, &overall_tcomm, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomp, &overall_tcomp, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    *tcomm_out = tcomm;
    *tcomp_out = tcomp;

    DEBUG_RANK(rank, "end!");

out:
    if (local_arr) free(local_arr);
    return ret;
}


int MPI_Is_sorted(int n, double *array, int *result, int root, MPI_Comm comm)
{
    int rank, size;
    int ret = MPI_SUCCESS;
    int answer = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double *firsts = (double *)calloc(size, sizeof(*firsts));
    double *lasts = (double *)calloc(size, sizeof(*firsts));

    ret = MPI_Gather(&array[0], 1, MPI_DOUBLE, firsts, 1, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Gather(&array[n-1], 1, MPI_DOUBLE, lasts, 1, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    if (rank == root) {
        for (int i = 0; i < size-1; i++) {
            if (lasts[i] > firsts[i+1]) {
                answer = 0;
                break;
            }
        }
    }

    ret = MPI_Bcast(&answer, 1, MPI_INT, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    *result = answer;

out:
    if (firsts) free(firsts);
    if (lasts) free(lasts);
    return ret;
}

int MPI_Exchange(int      n,
                 double  *array,
                 int      rank1,
                 int      rank2,
                 MPI_Comm comm,
                 double  *tcomm_out,
                 double  *tcomp_out)
{
    int rank, size, i, tag1 = 0, tag2 = 1;
    int ret = MPI_SUCCESS;
    double * b = (double *) calloc(n, sizeof(double));
    double * c;
    double overall_tcomm, overall_tcomp;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time  = 0.0f;
   
    MPI_Status status;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
 
    if(rank == rank1)
    {
        DEBUG_RANK(rank, "RANK1");
        time = MPI_Wtime();
        ret = MPI_Send(&array[0], n, MPI_DOUBLE, rank2, tag1, comm);
        tcomm += MPI_Wtime() - time; 
        if (ret != MPI_SUCCESS) goto out;
        DEBUG_RANK(rank, "RANK1 send");

        time = MPI_Wtime();
        ret = MPI_Recv(&b[0], n, MPI_DOUBLE, rank2, tag2, comm, &status);
        tcomm += MPI_Wtime() - time;
        if (ret != MPI_SUCCESS) goto out;
        DEBUG_RANK(rank, "RANK1 recv");

        time = MPI_Wtime();
        c = merge_array(n, array, n, b);
        for(i = 0; i < n; i++)
            array[i] = c[i];
        tcomp += MPI_Wtime() - time;
        DEBUG_RANK(rank, "RANK1 merge");

    }
    else if(rank == rank2)
    {
        DEBUG_RANK(rank, "RANK2");
        time = MPI_Wtime();
        ret = MPI_Recv(&b[0], n, MPI_DOUBLE, rank1, tag1, comm, &status);
        tcomm += MPI_Wtime() - time;
        if (ret != MPI_SUCCESS) goto out;
        DEBUG_RANK(rank, "RANK1 recv");

        time = MPI_Wtime();
        ret = MPI_Send(&array[0], n, MPI_DOUBLE, rank1, tag2, comm) ;
        tcomm += MPI_Wtime() - time;
        if (ret != MPI_SUCCESS) goto out;
        DEBUG_RANK(rank, "RANK2 send");

        time = MPI_Wtime();
        c = merge_array(n, array, n, b);
        for(i =0; i < n; i++)
            array[i] = c[i + n];
        tcomp += MPI_Wtime() - time;
        DEBUG_RANK(rank, "RANK1 merge");
    }

    /*
     * assumes that tcomm and tcomp passed in are being actively used, so we
     * just accumulate here rather than overwrite.
     */
    *tcomm_out += tcomm;
    *tcomp_out += tcomp;

out:
    if (ret != MPI_SUCCESS) LOG("Exchange error");
    if (b) free(b);
    return ret;
}

