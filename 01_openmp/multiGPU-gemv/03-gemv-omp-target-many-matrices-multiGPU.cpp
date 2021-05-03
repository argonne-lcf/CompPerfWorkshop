#define N 4096
#include <vector>
#include <omp.h>
#include "timer.h"

template<typename T>
void gemv(int deviceID, int n, T alpha, const T* __restrict__ A, const T* __restrict__ V, T* __restrict__ Vout)
{
#pragma omp target teams distribute map(to : A[:n * n], V[:n]) map(from : Vout[:n]) device(deviceID)
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
T* allocate(int deviceID, size_t n)
{
  T* ptr = new T[n];
  std::fill_n(ptr, n, T(1));
#pragma omp target enter data map(to : ptr[:n]) device(deviceID)
  return ptr;
}

template<class T>
void deallocate(int deviceID, T* ptr, size_t n)
{
#pragma omp target exit data map(delete : ptr[:n]) device(deviceID)
  delete[] ptr;
}

int main()
{
  const int num_devices = omp_get_num_devices();
  std::cout << "Found " << num_devices << " devices." << std::endl;

  std::vector<float*> manyA;
  std::vector<float*> manyV;
  std::vector<float*> manyVout;

  const int Num_calc = 8;
  for (int i = 0; i < Num_calc; i++)
  {
    manyA.push_back(allocate<float>(i % num_devices, N * N));
    manyV.push_back(allocate<float>(i % num_devices, N));
    manyVout.push_back(allocate<float>(i % num_devices, N));
  }

  {
    Timer local("multiGEMV");
#pragma omp parallel for
    for (int i = 0; i < Num_calc; i++)
      gemv(i % num_devices, N, 1.0f, manyA[i], manyV[i], manyVout[i]);
  }

  for (int i = 0; i < Num_calc; i++)
  {
    auto* __restrict__ Vout = manyVout[i];
#pragma omp target update from(Vout[:N]) device(i % num_devices)
    for (int j = 0; j < N; j++)
      if (Vout[j] != N)
      {
        std::cerr << "Calculation " << i << " Vout[" << j << "] != " << N << ", wrong value is " << Vout[j]
                  << std::endl;
        break;
      }

    deallocate(i % num_devices, manyA[i], N * N);
    deallocate(i % num_devices, manyV[i], N);
    deallocate(i % num_devices, manyVout[i], N);
  }
}
