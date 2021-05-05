program main

  implicit none

  integer, parameter :: N = 1000000
  double precision, parameter :: a = 3.0
  double precision, allocatable :: x(:), y(:)
  integer :: i

  allocate(x(N))
  allocate(y(N))

  do i = 1, N
     x(i) = 1.0 * i
     y(i) = 2.0 * i
  end do

  !$acc parallel loop
  do i = 1, N
     y(i) = a * x(i) + y(i)
  end do

end program main
