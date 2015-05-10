/*
 * solver.cpp
 *
 *  Created on: Feb 19, 2015
 *      Author: dimalit
 */

#include "solver.h"
#include <cassert>

// externally-visible:
double init_step;
double tolerance;
int m;
double n;
double theta_e, delta_e, r_e, gamma_0_2;
double a0;

const double PI = 4*atan(1.0);
//now we havee a0 in state at protobuf: double *array_a0;					// needed for RHS

PetscErrorCode RHSFunction_te(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode RHSFunction_tm(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode wrap_function(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);
//void vec_to_a0(Vec v);

PetscErrorCode solve_abstract(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time, PetscErrorCode (*rhs_function)(TS ts, PetscReal t,Vec in,Vec out,void*));

PetscErrorCode solve_te(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time)
{
	return solve_abstract(initial_state, max_steps, max_time,
			   	   	   	  out_state, out_rhs, out_steps, out_time, RHSFunction_te);
}

PetscErrorCode solve_tm(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time)
{
	return solve_abstract(initial_state, max_steps, max_time,
			   	   	   	  out_state, out_rhs, out_steps, out_time, RHSFunction_tm);
}

PetscErrorCode solve_abstract(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time, PetscErrorCode (*rhs_function)(TS ts, PetscReal t,Vec in,Vec out,void*))
{
	fprintf(stderr, "+");
	fflush(stderr);

	PetscErrorCode ierr;

	int lo, hi;
	VecGetOwnershipRange(initial_state, &lo, &hi);

//	array_a0 = new PetscScalar[hi - lo];
//	vec_to_a0(initial_state);

	TS ts;
	ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
	ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

	ierr = TSSetRHSFunction(ts, NULL, rhs_function, 0);CHKERRQ(ierr);

//	ierr = TSSetPostStep(ts, wrap_function);CHKERRQ(ierr);
//!!!	ierr = TSMonitorSet(ts, wrap_function, NULL, NULL);CHKERRQ(ierr);

	ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
	ierr = TSRKSetType(ts, TSRK4);CHKERRQ(ierr);

	ierr = TSSetInitialTimeStep(ts, 0.0, init_step);CHKERRQ(ierr);
	ierr = TSSetTolerances(ts, tolerance, NULL, tolerance, NULL);CHKERRQ(ierr);
	ierr = TSSetDuration(ts, max_steps, max_time);CHKERRQ(ierr);

	ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

	ierr = TSSolve(ts, initial_state);CHKERRQ(ierr);

//	delete[] array_a0;

	// get final state
	Vec out;
	ierr = TSGetSolution(ts, &out);CHKERRQ(ierr);
	ierr = VecDuplicate(out, out_state);CHKERRQ(ierr);
	ierr = VecCopy(out, *out_state);CHKERRQ(ierr);

	// get final step and time
	TSGetTime(ts, out_time);
	TSGetTimeStepNumber(ts, out_steps);

	// get final RHS
	ierr = VecDuplicate(*out_state, out_rhs);CHKERRQ(ierr);			// use same dimensions
	rhs_function(ts, *out_time, *out_state, *out_rhs, NULL);

	double tstep;
	TSGetTimeStep(ts, &tstep);
	fprintf(stderr, "%lf", tstep);
	fflush(stderr);

	TSDestroy(&ts);

	fprintf(stderr, "- ");
	fflush(stderr);

	return 0;
}

// compute BesselJ_n_'(x)
double Jn(double x){
	return jn(n, x);
//	return 0.5*n*x;
}

// compute BesselJ_n(x)
double Jn_(double x){
	return jn(n-1, x) - n * jn(n, x)/x;
//	return 0.5*n;
}

PetscErrorCode RHSFunction_te(TS ts, PetscReal t,Vec in,Vec out,void*){
	PetscErrorCode ierr;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double E_e, phi_e;

	// bcast E and phi
	if(rank == 0){
		const double* data;
		ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
		E_e =data[0];
		phi_e = data[1];
		ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(in, &lo, &hi);
	if(lo < 2)	// exclude E, phi
		lo = 2;

	double sum_sin = 0;
	double sum_cos = 0;
	for(int i=lo; i<hi; i+=3){
		double nak[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, nak);
		sum_sin += nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		sum_cos += nak[1]*Jn_(nak[1])*cos(2*PI*nak[2]+phi_e);
	}

	double tmp = 0;
	MPI_Reduce(&sum_sin, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_sin = tmp;
	tmp = 0;
	MPI_Reduce(&sum_cos, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_cos = tmp;

	// compute derivatives of E_e and phi_e
	double dE, dphi;
	if(rank == 0){
		dE = theta_e*E_e + 1.0/m*sum_sin;
		dphi = ( delta_e*E_e + 1.0/m*sum_cos ) / E_e;
		dphi = 0.0;//!!!
			VecSetValue(out, 0, dE, INSERT_VALUES);
			VecSetValue(out, 1, dphi, INSERT_VALUES);
		VecAssemblyBegin(out);
		VecAssemblyEnd(out);
	}
	MPI_Bcast(&dE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dphi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// compute n, a, k
	VecGetOwnershipRange(out, &lo, &hi);
	if(lo < 2)
		lo = 2;

	for(int i=lo; i<hi; i+=3){
		double nak[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, nak);

//		double dn = -r_e*E_e*nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
//		double da = -n*E_e*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
//		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1])-n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*sin(2*PI*nak[2]+phi_e);

		double dn = 0;//-r_e*nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double da = -sin(2*PI*nak[2]+phi_e);
		double dk = -1.0 + 1.0/nak[1]*sin(2*PI*nak[2]+phi_e);

		//another version that behaves the same way as TM when n=-1
		//double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1])+n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*cos(2*PI*nak[2]+phi_e);
			dk /= 2*PI;

		// compute full sin arg and limit it to [-PI, PI]
		double arg = 2*PI*nak[2]+phi_e;
		arg = fmod(arg, 2*PI);
		if(arg > PI)
			arg -= 2*PI;

		if(fabs(sin(arg)) < 0.01){
			//dk = dphi/2.0/PI;
			//fprintf(stderr, "p[%d] = %lf\n", (i-lo)/3, sin(arg));
		}// sin < 0.01

		VecSetValue(out, i, dn, INSERT_VALUES);
		VecSetValue(out, i+1, da, INSERT_VALUES);
		VecSetValue(out, i+2, dk, INSERT_VALUES);

		fprintf(stderr, "a:%lf\tksi:%lf\n", nak[1], nak[2]);
		fprintf(stderr, "da:%lf\tdk:%lf\n", da, dk);
		fflush(stderr);
	}


	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);
//	VecView(out, PETSC_VIEWER_STDERR_WORLD);

	return 0;
}

PetscErrorCode RHSFunction_tm(TS ts, PetscReal t,Vec in,Vec out,void*){
	PetscErrorCode ierr;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//	static int counter = 0;
//	static double prev_t = 0.0;
//	counter++;
//	if(rank==0){
//		fprintf(stderr, "RHS count = %d dt = %lf\n", counter, t-prev_t);
//		fflush(stderr);
//		prev_t = t;
//	}

	double E_e, phi_e;

	// bcast E and phi
	if(rank == 0){
		const double* data;
		ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
		E_e = data[0];
		phi_e = data[1];
		ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(in, &lo, &hi);
	if(lo < 2)	// exclude E, phi
		lo = 2;

	double sum_sin = 0;
	double sum_cos = 0;
	for(int i=lo; i<hi; i+=3){
		double nak[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, nak);
		sum_sin += Jn(nak[1])*sin(2*PI*nak[2]+phi_e);
		sum_cos += Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
	}

	double tmp = 0;
	MPI_Reduce(&sum_sin, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_sin = tmp;
	tmp = 0;
	MPI_Reduce(&sum_cos, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_cos = tmp;

	// compute derivatives of E_e and phi_e
	if(rank == 0){
		// RESOLVED: here appeared 2.0 and book should have E=E/sqrtR (without 2)
		double dE = theta_e*E_e + 1.0/m*sum_cos;
		double dphi = ( delta_e*E_e - 1.0/m*sum_sin ) / E_e;
		//double dE = theta_e*E_e + 2.0/m*sum_cos;
		//double dphi = (delta_e*E_e - 2.0/m*sum_sin) / E_e;
			VecSetValue(out, 0, dE, INSERT_VALUES);
			VecSetValue(out, 1, dphi, INSERT_VALUES);
		VecAssemblyBegin(out);
		VecAssemblyEnd(out);
	}

	// compute n, a, k
	VecGetOwnershipRange(out, &lo, &hi);
	if(lo < 2)
		lo = 2;

	for(int i=lo; i<hi; i+=3){
		double nak[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, nak);

//		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1]) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double dn = -r_e*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
		double da = -n/nak[1]*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1]) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
			dk /= 2*PI;

		VecSetValue(out, i, dn, INSERT_VALUES);
		VecSetValue(out, i+1, da, INSERT_VALUES);
		VecSetValue(out, i+2, dk, INSERT_VALUES);
	}

	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);
//	VecView(out, PETSC_VIEWER_STDERR_WORLD);

	return 0;
}

void wrap_ksi_in_vec(Vec u){
	int begin, end;
	VecGetOwnershipRange(u, &begin, &end);

	double* arr;
	VecGetArray(u, &arr);

	for(int i=0; i<(end-begin); i++){
		// skip E, phi
		if(begin+i <= 1)
			continue;

		// need ksi only
		if((begin+i-2) % 3 == 2){
			//fprintf(stderr, "%lf:", arr[i]);
			if(arr[i] > 0.5)
				arr[i] -= 1.0;
			else if(arr[i] < -0.5)
				arr[i] += 1.0;
			//fprintf(stderr, "%lf ", arr[i]);
		}// if ksi
	}
	//fprintf(stderr, "\n");
	VecRestoreArray(u, &arr);
}

// NOTE: this doesn't work in TSPostStep - because solution is rewritten after call to this func
// use it in TSMonitor
PetscErrorCode wrap_function(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx){
//	Vec u;
//	TSGetSolution(ts, &u);
	wrap_ksi_in_vec(u);

	return 0;
	// and consider small a also

	double* arr;
	VecGetArray(u, &arr);

	int lo, hi;
	VecGetOwnershipRange(u, &lo, &hi);

	double E_e, phi_e;
	if(lo == 0){
		E_e = arr[0];
		phi_e = arr[1];
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	int i=0;
	if(lo == 0)	// exclude E, phi
		i = 2;
	for(; i<(hi-lo); i+=3){
		double& n = arr[i+0];
		double& a = arr[i+1];
		double& k = arr[i+2];

		if(a > 0.001)
			continue;

		// compute full sin arg and limit it to [-PI, PI]
		double arg = 2*PI*k+phi_e;
		arg = fmod(arg, 2*PI);
		if(arg > PI)
			arg -= 2*PI;

		if(sin(arg) < 1e-6)
			continue;

		// go to small sin
		double right_zero, left_zero;
		if(arg < 0){
			right_zero = 0.0;
			left_zero = -PI;
		}
		else if(arg > 0){
			right_zero = PI;
			left_zero = 0.0;
		}
		else{
			assert(false);			// shouldn't be because sin < 0.001 we already considered
		}

		double dir;
		if(n>0)
			dir = sin(arg)>0 ? 1 : -1;
		else
			dir = sin(arg)>0 ? -1 : 1;

		if(dir > 0){
			k += (right_zero - arg)/2/PI;
		}
		else{
			k += (left_zero - arg)/2/PI;
		}// dir r or l
	}// for

	VecRestoreArray(u, &arr);

	return 0;

}

//void vec_to_a0(Vec v){
//	double* arr;
//	VecGetArray(v, &arr);
//
//	int rank;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//	int localsize;
//	VecGetLocalSize(v, &localsize);
//
//	int shift = 0;
//	if(rank == 0)
//		shift = 2;
//
//	for(int i=0; i<localsize-shift; i++){
//		array_a0[i] = arr[i+shift];
//		fprintf(stderr, "Putting to %d %lf\n", i, arr[i+shift]);
//	}
//
//	VecRestoreArray(v, &arr);
//}

