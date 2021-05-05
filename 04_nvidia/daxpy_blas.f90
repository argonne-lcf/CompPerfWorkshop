program main

  use cublas

  implicit none

  integer, parameter :: N = 1000000
  double precision, allocatable, managed :: a
  double precision, allocatable, managed :: x(:), y(:)
  integer :: i

  allocate(a)
  a = 3.0

  allocate(x(N))
  allocate(y(N))

  do i = 1, N
     x(i) = 1.0 * i
     y(i) = 2.0 * i
  end do

  call cublasDaxpy(N, a, x, 1, y, 1)

end program main
