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

int MPI_Sort_direct(int n, double *array, int root, MPI_Comm comm,    
                    double *out_tcomm, double *out_tcomp);

int MPI_Sort_bucket(int n, double *array, int root, MPI_Comm comm,
                    double *out_tcomm, double *out_tcomp);

int MPI_Sort_odd_even(int n, double *array, int root, MPI_Comm comm,
                      double *out_tcomm, double *out_tcomp);

int MPI_Sort_shell(int n, double *array, int root, MPI_Comm comm,
                   double *out_tcomm, double *out_tcomp);

int MPI_Sort_bitonic(int n, double *array, int root, MPI_Comm comm,
                     double *out_tcomm, double *out_tcomp);


int MPI_Exchange(int n, double * array, int rank1, int rank2, MPI_Comm comm,
                 double *out_tcomm, double *out_tcomp);

int MPI_Is_sorted(int n, double *array, int *result,
                  int root, MPI_Comm comm);

typedef enum sort_kind_e {
    SORT_DIRECT,
    SORT_BUCKET,
    SORT_ODD_EVEN,
    SORT_SHELL,
    SORT_BITONIC,
    _SORT_COUNT,
} sort_kind_t;

static inline const char*
sort_kind_to_str(sort_kind_t kind)
{
    switch(kind) {
        case SORT_DIRECT:   return "DIRECT SORT";
        case SORT_BUCKET:   return "BUCKET SORT";
        case SORT_ODD_EVEN: return "ODD_EVEN SORT";
        case SORT_SHELL:    return "SHELL SORT";
        case SORT_BITONIC:  return "BITONIC SORT";
        default:            return "UNKNOWN";
    }
}

typedef int (*MPI_Sort_fn)(int n, double *array, int root, MPI_Comm comm,
                           double *out_tcomm, double *out_tcomp);

const MPI_Sort_fn MPI_Sort_funcs[_SORT_COUNT] = {
    MPI_Sort_direct,
    MPI_Sort_bucket,
    MPI_Sort_odd_even,
    MPI_Sort_shell,
    MPI_Sort_bitonic,
};

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

    if (argc < 2) {
        LOG("not enough arguments");
        return 1;
    }

    int argv_int = atoi(argv[1]);
    if (argv_int < 0 || argv_int >= _SORT_COUNT) {
        LOG("sort kind out of bounds: %d", argv_int);
        LOG("setting sort: %s", sort_kind_to_str(_SORT_COUNT-1));
        argv_int = _SORT_COUNT-1;
    }

    sort_kind_t sort_kind = (sort_kind_t)argv_int;

	MPI_Init(&argc, &argv);
   	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   	MPI_Comm_size(MPI_COMM_WORLD, &size);

    array = (double *) calloc(n, sizeof(double));
	if(rank == 0)
	{
	   srand(((unsigned)time(NULL)+rank));

	   for(i = 0; i < n; i++) {
	      array[i]=((double)rand()/RAND_MAX)*m;
	   }

	}

    double tcomm, tcomp;

    ret = MPI_Sort_funcs[sort_kind](n, array, 0, MPI_COMM_WORLD, &tcomm, &tcomp); 

    if (ret != MPI_SUCCESS) {
        LOG("MPI failure: %d", ret);
        goto final;
    }
    
    if (rank == 0) {
        LOG("\n%s", sort_kind_to_str(sort_kind));
        LOG("Comm Time: %lf Comp Time: %lf Total: %lf",
            rank, tcomm, tcomp, tcomm + tcomp);
    }

final:
	MPI_Finalize();
}

double*
merge_array(int n, double * a, int m, double * b)
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

void
merge_sort(int n, double * a)
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

void
bubble_sort(int n, double * a)
{
    for(int i=0;i<n-1;i++) {
        for(int j=0;j<n-1;j++) {
            if(a[j]>a[j+1])swap(a+j, a+j+1);
        }
    }
}

// swap two doubles
void
swap (double * a, double * b)
{
   double temp;

   temp=*a;*a=*b;*b=temp;
}

int
MPI_Sort_direct(int n,
                double *array,
                int root,
                MPI_Comm comm,
                double *out_tcomm,
                double *out_tcomp)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int ret = MPI_SUCCESS;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time  = 0.0f;
    double overall_tcomm, overall_tcomp;
    
    double *local_array = (double *)calloc(n/size, sizeof(double));
    if (!local_array) goto out;
    
    // scatter array to local_array with double elements
    time = MPI_Wtime();
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_array, n/size, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    // sequential sort local_array
    time = MPI_Wtime();
    bubble_sort(n/size, local_array);
    time = MPI_Wtime() - time;
    tcomp += time;

    // gather local_array to array with double elements
    time = MPI_Wtime();
    ret = MPI_Gather(local_array, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    if(rank == 0) {
        time = MPI_Wtime();
        // merge the size chunks of array
        for(int i=1;i<size;i++) {
            double *tmp_array = merge_array(n/size*i, array, n/size, array+n/size*i);
            for(int j=0;j<(i+1)*n/size;j++)array[j] = tmp_array[j];
        }
        time = MPI_Wtime() - time;
        tcomp += time;
    }

    ret = MPI_Reduce(&tcomm, &overall_tcomm, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomp, &overall_tcomp, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;
    
    *out_tcomm = overall_tcomm;
    *out_tcomp = overall_tcomp;

out:
    if (local_array) free(local_array);
    return ret;
}

#ifndef MPI_SORT_BUCKET_MAX
    #define MPI_SORT_BUCKET_MAX 5
#endif //MPI_SORT_BUCKET_MAX

int
MPI_Sort_bucket(int n,
                double *array,
                int root,
                MPI_Comm comm,
                double *out_tcomm,
                double *out_tcomp)
{
    int     rank, size;
    double *bucket;
    int     bucket_count;
    int    *recv_counts, *displacements;
    int     ret = MPI_SUCCESS;
    double  tcomm = 0.0f;
    double  tcomp = 0.0f;
    double  time;
    double  overall_tcomm, overall_tcomp;
    
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
    time = MPI_Wtime();
    ret = MPI_Bcast(array, n, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    /* 
     * collect as elems to proc ranks bucket
     * since we don;t know how big the bucket could be we init to n
     */
    time = MPI_Wtime();
    for (int i = 0; i < n; i++) {
        /* check if a[i] is in the bucket [max/size*rank, max/size*(rank+1)] */
        if (array[i] >= MPI_SORT_BUCKET_MAX/size+rank && array[i] < MPI_SORT_BUCKET_MAX/size*(rank+1)) {
            bucket[bucket_count++] = array[i];
        }
    }
    merge_sort(bucket_count, bucket);
    time = MPI_Wtime() - time;
    tcomp += time;

    /* gatherv */
    time = MPI_Wtime();
    ret = MPI_Gather(&bucket_count, 1, MPI_INT, recv_counts, 1, MPI_INT, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    time = MPI_Wtime();
    if (rank == root) {
        displacements[0] = 0;
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }
    time = MPI_Wtime() - time;
    tcomp += time;

    time = MPI_Wtime();
    ret = MPI_Gatherv(bucket, bucket_count, MPI_DOUBLE,
                      array, recv_counts, displacements, MPI_DOUBLE,
                      root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomm, &overall_tcomm, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomp, &overall_tcomp, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    *out_tcomm = overall_tcomm;
    *out_tcomp = overall_tcomp;

out:
    if (bucket) free(bucket);
    return ret;
}

int
MPI_Sort_odd_even(int n,
                  double *array,
                  int root,
                  MPI_Comm comm,
                  double *out_tcomm,
                  double *out_tcomp)
{
    int rank, size;
    int ret = MPI_SUCCESS;
    double tcomm = 0.0f;
    double tcomp = 0.0f;
    double time;
    double overall_tcomm, overall_tcomp;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    double *local_arr = (double *)calloc(n/size, sizeof(*local_arr));
    if (!local_arr) {
        ret = MPI_T_ERR_MEMORY;
        goto out;
    }
  
    time = MPI_Wtime();
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_arr, n/size, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    time = MPI_Wtime();
    merge_sort(n/size, local_arr);
    time = MPI_Wtime() - time;
    tcomp += time;

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
        time = MPI_Wtime();
        ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
        time = MPI_Wtime() - time;
        tcomm += time;
        if (ret != MPI_SUCCESS) goto out;

        if (is_sorted) {
            //LOG("number of steps: %d", step);
            break;
        }
    }
    
    time = MPI_Wtime();
    ret = MPI_Gather(local_arr, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;
    
    ret = MPI_Reduce(&tcomm, &overall_tcomm, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomp, &overall_tcomp, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    *out_tcomm = overall_tcomm;
    *out_tcomp = overall_tcomp;

out:
    if (local_arr) free(local_arr);

    return ret;
}
 
int
MPI_Sort_shell(int n,
               double *array,
               int root,
               MPI_Comm comm,
               double *out_tcomm,
               double *out_tcomp)
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
  
    time = MPI_Wtime();
    ret = MPI_Scatter(array, n/size, MPI_DOUBLE, local_arr, n/size, MPI_DOUBLE, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
    if (ret != MPI_SUCCESS) goto out;

    time = MPI_Wtime();
    merge_sort(n/size, local_arr);
    time = MPI_Wtime() - time;
    tcomm += time;

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
    time = MPI_Wtime();
    ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
    time = MPI_Wtime() - time;
    tcomm += time;
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

        time = MPI_Wtime();
        ret = MPI_Is_sorted(n/size, local_arr, &is_sorted, root, comm);
        time = MPI_Wtime() - time;
        tcomm += time;
        if (ret != MPI_SUCCESS) goto out;

        if (is_sorted) {
            // LOG("number of steps: %d", step);
            break;
        }
    }

gather:
    ret = MPI_Gather(local_arr, n/size, MPI_DOUBLE, array, n/size, MPI_DOUBLE, root, comm);
    if (ret != MPI_SUCCESS) goto out;
    
    ret = MPI_Reduce(&tcomm, &overall_tcomm, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    ret = MPI_Reduce(&tcomp, &overall_tcomp, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (ret != MPI_SUCCESS) goto out;

    *out_tcomm = overall_tcomm;
    *out_tcomp = overall_tcomp;

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

int
MPI_Sort_bitonic(int n,
                 double  *array,
                 int      root,
                 MPI_Comm comm,
                 double  *out_tcomm,
                 double  *out_tcomp)
{
    int rank, size;
    int ret = MPI_SUCCESS;
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

    *out_tcomm = overall_tcomm;
    *out_tcomp = overall_tcomp;

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
                 double  *out_tcomm,
                 double  *out_tcomp)
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
    *out_tcomm += tcomm;
    *out_tcomp += tcomp;

out:
    if (ret != MPI_SUCCESS) LOG("Exchange error");
    if (b) free(b);
    return ret;
}

