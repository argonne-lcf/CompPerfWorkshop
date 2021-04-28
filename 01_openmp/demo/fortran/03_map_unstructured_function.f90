module data
  double precision, allocatable :: a(:) 
  double precision, allocatable :: b(:) 
  double precision, allocatable :: c(:) 
end module data

subroutine daxpy( scalar, num_elements )
  use data
  implicit none
  integer num_elements, j
  double precision scalar

!$omp target teams distribute parallel do simd map(tofrom:a) map(to:b)
    do j=1,num_elements
       a(j) = a(j)+scalar*b(j)
    end do
!$omp end target teams distribute parallel do simd

end subroutine daxpy

program main
  use data
  implicit none
  double precision  scalar
  integer err, j
  integer num_errors
  integer num_elements

  scalar = 8d0
  num_errors = 0
  num_elements = 1024

  allocate (a(num_elements),stat=err)
  allocate (b(num_elements),stat=err)
  allocate (c(num_elements),stat=err)

  ! initialize on the host
  do j=1,num_elements
     a(j) = 0d0
     b(j) = j
     c(j) = 0d0
  end do

!$omp target enter data map(to:a)
!$omp target enter data map(to:b)
!$omp target enter data map(to:c)

  call daxpy( a, b, scalar, num_elements )

  call daxpy( c, a, scalar, num_elements )

!$omp target update from(c)

  ! error checking
  do j=1,num_elements
     if( c(j) - j*scalar*scalar .gt. 0.000001 ) then
        num_errors = num_errors + 1
     end if

  end do

!$omp target exit data map(release:a)
!$omp target exit data map(release:b)
!$omp target exit data map(release:c)

  deallocate(a);
  deallocate(b);
  deallocate(c);

  if(num_errors == 0) write(*,*) "Success!\n"

end program main

