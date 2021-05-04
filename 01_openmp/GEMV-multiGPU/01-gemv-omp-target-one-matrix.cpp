#define N 8192
#include "timer.h"

template<typename T>
void gemv(int n, T alpha, const T* __restrict__ A, const T* __restrict__ V, T* __restrict__ Vout)
{
#pragma omp target teams distribute map(to : A[:n * n], V[:n]) map(from : Vout[:n])
  for (int row = 0; row < n; row++)
  {
    T sum                       = T(0);
    const T* __restrict__ A_row = A + row * n;
#pragma omp parallel for reduction(+ : sum)
    for (int col = 0; col < n; col++)
      sum += A_row[col] * V[col];
    Vout[row] = sum * alpha;
  }
}

template<class T>
T* allocate(size_t n)
{
  T* ptr = new T[n];
  std::fill_n(ptr, n, T(1));
#pragma omp target enter data map(to : ptr[:n])
  return ptr;
}

template<class T>
void deallocate(T* ptr, size_t n)
{
#pragma omp target exit data map(delete : ptr[:n])
  delete[] ptr;
}

int main()
{
  auto* A    = allocate<float>(N * N);
  auto* V    = allocate<float>(N);
  auto* Vout = allocate<float>(N);

  {
    Timer local("GEMV");
    gemv(N, 1.0f, A, V, Vout);
  }

#pragma omp target update from(Vout[:N])
  for (int i = 0; i < N; i++)
    if (Vout[i] != N)
    {
      std::cerr << "Vout[" << i << "] != " << N << ", wrong value is " << Vout[i] << std::endl;
      break;
    }

  deallocate(A, N * N);
  deallocate(V, N);
  deallocate(Vout, N);
}
