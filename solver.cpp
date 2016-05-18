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
double atolerance, rtolerance;
bool use_ifunction;
bool use_ijacobian;
int m;
double n;
double theta_e, delta_e, r_e, gamma_0_2;
double a0;
double alpha;	// may be used to tie eta to a: eta=-alpha*a

const double PI = 4*atan(1.0);
//now we have a0 in state at protobuf: double *array_a0;					// needed for RHS

PetscErrorCode RHSFunction_te(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode RHSFunction_tm(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode wrap_function(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

PetscErrorCode ifunction_te(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx);
PetscErrorCode ijacobian_te(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift, Mat A, Mat B,void *ctx);
PetscErrorCode ifunction_tm(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx);
PetscErrorCode ijacobian_tm(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift, Mat A, Mat B,void *ctx);

PetscErrorCode solve_abstract(Vec initial_state, TSRHSFunction* rhs_function,
		   TSIFunction ifunction, TSIJacobian ijacobian, int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time));
PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

PetscErrorCode solve_tm(Vec initial_state, int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time))
{
	if(use_ifunction)
		return solve_abstract(initial_state, (TSRHSFunction*)RHSFunction_tm, ifunction_tm, ijacobian_tm, max_steps, max_time, step_func);
	else
		return solve_abstract(initial_state, (TSRHSFunction*)RHSFunction_tm, NULL, NULL, max_steps, max_time, step_func);
}

PetscErrorCode solve_te(Vec initial_state, int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time))
{
	if(use_ifunction)
		return solve_abstract(initial_state, (TSRHSFunction*)RHSFunction_te, ifunction_te, ijacobian_te, max_steps, max_time, step_func);
	else
		return solve_abstract(initial_state, (TSRHSFunction*)RHSFunction_te, NULL, NULL, max_steps, max_time, step_func);
}

PetscErrorCode solve_abstract(Vec initial_state, TSRHSFunction* rhs_function,
		   TSIFunction ifunction, TSIJacobian ijacobian, int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time))
{
//	VecView(initial_state, PETSC_VIEWER_STDERR_WORLD);

	PetscErrorCode ierr;

	int lo, hi;
	VecGetOwnershipRange(initial_state, &lo, &hi);

//	array_a0 = new PetscScalar[hi - lo];
//	vec_to_a0(initial_state);

	TS ts;
	ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
	ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

	if(!ifunction){
		ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
		ierr = TSRKSetType(ts, TSRK4);CHKERRQ(ierr);
		// XXX: strange cast - should work without it too!
		ierr = TSSetRHSFunction(ts, NULL, (PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))rhs_function, 0);CHKERRQ(ierr);
	}
	else{
		ierr = TSSetType(ts, TSROSW);CHKERRQ(ierr);
		ierr = TSSetIFunction(ts, NULL, ifunction, NULL);CHKERRQ(ierr);

		Mat A;
		MatCreate(PETSC_COMM_WORLD,&A);
		MatSetSizes(A,2+3*m,2+3*m,PETSC_DETERMINE,PETSC_DETERMINE);
		MatSetFromOptions(A);
		MatSetUp(A);
		ierr = TSSetIJacobian(ts, A, A, ijacobian, NULL);CHKERRQ(ierr);

		//!!! dangerous!
		TSSetMaxStepRejections(ts, 30);
		TSSetMaxSNESFailures(ts, -1);
	}

//	ierr = TSSetPostStep(ts, wrap_function);CHKERRQ(ierr);
	ierr = TSMonitorSet(ts, wrap_function, NULL, NULL);CHKERRQ(ierr);

	ierr = TSSetInitialTimeStep(ts, 0.0, init_step);CHKERRQ(ierr);
	ierr = TSSetTolerances(ts, atolerance, NULL, rtolerance, NULL);CHKERRQ(ierr);
//	fprintf(stderr, "steps=%d time=%lf ", max_steps, max_time);

	ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

	ierr = TSSetSolution(ts, initial_state);CHKERRQ(ierr);

	ierr = TSSetDuration(ts, max_steps, max_time);CHKERRQ(ierr);

	ierr = TSMonitorSet(ts, step_monitor, (void*)step_func, NULL);
	ierr = TSSolve(ts, initial_state);CHKERRQ(ierr);			// results are "returned" in step_monitor

	double tstep;
	TSGetTimeStep(ts, &tstep);

	TSDestroy(&ts);
	return 0;
}

PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx){
	bool (*step_func)(Vec state, Vec rhs, int steps, double time) = (bool (*)(Vec state, Vec rhs, int steps, double time)) mctx;

	PetscErrorCode ierr;

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);

	// get final RHS
	Vec rhs;
	TSRHSFunction func;
	ierr = TSGetRHSFunction(ts, &rhs, &func, NULL);CHKERRQ(ierr);
	func(ts, time, u, rhs, NULL);	// XXX: why I need to call func instead of getting rhs from TSGetRhsFunction??

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);
//	VecView(rhs, PETSC_VIEWER_STDERR_WORLD);

	PetscInt true_steps;
	TSGetTimeStepNumber(ts, &true_steps);

	bool res = step_func(u, rhs, true_steps, time);
	if(!res){
		//TSSetConvergedReason(ts, TS_CONVERGED_USER);
		TSSetDuration(ts, steps, time);
	}

	return 0;
}

// compute BesselJ_n(x)
double Jn(double x){
	return jn(n, x);
//	return 0.5*n*x;
}

// compute BesselJ_n'(x)
double Jn_(double x){
// prev:	return jn(n-1, x) - n * jn(n, x)/x;
// x->0:	return 0.5*n;
	return 0.5*(jn(n-1, x) - jn(n+1, x));
}

double Jn_2(double x){
	return 0.25*(jn(n-2, x) - 2*jn(n, x) + jn(n+2, x));
}

PetscErrorCode RHSFunction_te(TS ts, PetscReal t,Vec in,Vec out,void*){
	//	fprintf(stderr, "%s\n", __FUNCTION__);
	PetscErrorCode ierr;

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double E_e, phi_e;

	// bcast E and phi
	const double* data;
	ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
	if(rank == 0){
		E_e =data[0];
		phi_e = data[1];
	}
	ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
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
			VecSetValue(out, 0, dE, INSERT_VALUES);
			VecSetValue(out, 1, dphi, INSERT_VALUES);
	}
	VecAssemblyBegin(out);
	VecAssemblyEnd(out);
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

		double dn = -r_e*E_e*nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double da = -n*E_e*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);


		//previous version - before simplifications - corresponds to Jacobians!!
		//double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1])+n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*cos(2*PI*nak[2]+phi_e);

		// simplified version - contradicts to jacobians!!
		// +nE was before, -nE got from Kuklin with simplyfied eqns
		double dk = nak[0]*(1-gamma_0_2) + n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*cos(2*PI*nak[2]+phi_e);

		// just show the problem with stiffness
//		double dn = 0;//-r_e*nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
//		double da = -sin(2*PI*nak[2]+phi_e);
//		double dk = -1.0 + 1.0/nak[1]*sin(2*PI*nak[2]+phi_e);

		//another version that behaves the same way as TM when n=-1
		//double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1])+n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*cos(2*PI*nak[2]+phi_e);

		dk /= 2*PI;

		VecSetValue(out, i, dn, INSERT_VALUES);
		VecSetValue(out, i+1, da, INSERT_VALUES);
		VecSetValue(out, i+2, dk, INSERT_VALUES);

//		fprintf(stderr, "a:%lf\tksi:%lf\n", nak[1], nak[2]);
//		fprintf(stderr, "da:%lf\tdk:%lf\n", da, dk);
//		fflush(stderr);
	}


	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);
//	VecView(out, PETSC_VIEWER_STDERR_WORLD);

	return 0;
}

// TODO ifunction and ijacobian probably won't work with n>=2. And thay are for previous equations!
PetscErrorCode ifunction_te(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	PetscScalar    *x,*xdot,*f;

	VecGetArray(X,&x);
	VecGetArray(Xdot,&xdot);
	VecGetArray(F,&f);

	double E_e, phi_e;

	// bcast E and phi
	if(rank == 0){
		E_e = x[0];
		phi_e = x[1];
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(X, &lo, &hi);

	double sum_sin = 0;
	double sum_cos = 0;
	for(int i=lo < 2? 2 : lo; i<hi; i+=3){
		double nak[3] = {x[i-lo], x[i+1-lo], x[i+2-lo]};
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
		f[0] = xdot[0] - dE;
		f[1] = xdot[1] - dphi;
	}
	MPI_Bcast(&dE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dphi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// compute n, a, k
	for(int i=lo<2? 2 : lo; i<hi; i+=3){
		double nak[3] = {x[i-lo], x[i+1-lo], x[i+2-lo]};

		double dn = -r_e*E_e*nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double da = -n*E_e*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1])+n*E_e*Jn(nak[1])*(1-n*n/nak[1]/nak[1])*cos(2*PI*nak[2]+phi_e);

		dk /= 2*PI;

		f[i-lo] = xdot[i-lo] - dn;
		f[i+1-lo] = xdot[i+1-lo] - da;
		f[i+2-lo] = xdot[i+2-lo] - dk;
	}

	VecRestoreArray(X,&x);
	VecRestoreArray(Xdot,&xdot);
	VecRestoreArray(F,&f);

	return 0;
}

PetscErrorCode ijacobian_te(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift, Mat A, Mat B,void *ctx){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	PetscScalar *x, *xdot;

	int lo, hi;
	MatGetOwnershipRange(B, &lo, &hi);

	VecGetArray(X, &x);
	VecGetArray(Xdot, &xdot);

	// 0 prepare
	double E_e, phi_e;
	double sum_sin = 0, sum_cos = 0;

	// bcast E and phi
	if(lo == 0){
		E_e = x[0];
		phi_e = x[1];
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	for(int i = lo < 2 ? 2 : lo; i<hi; i+=3){
		double a = x[i-lo+1];
		double k = x[i-lo+2];
		sum_sin += a*Jn_(a)*sin(2*PI*k+phi_e);
		sum_cos += a*Jn_(a)*cos(2*PI*k+phi_e);
	}
	double tmp = 0;
	MPI_Reduce(&sum_sin, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_sin = tmp;
	tmp = 0;
	MPI_Reduce(&sum_cos, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_cos = tmp;

	// 1 add a-term
	for(int i=lo; i<hi; i++){
		MatSetValue(B, i, i, shift, INSERT_VALUES);
	}

	// special case to test if we really need this jacobian
	if(use_ijacobian){

	// 2 compute E and phi
	if(lo==0){
		double Ee = theta_e;
		double Ephi = 1.0/m * sum_cos;

		MatSetValue(B, 0, 0, shift-Ee, INSERT_VALUES);
		MatSetValue(B, 0, 1, -Ephi, INSERT_VALUES);

		for(int j = 2; j<2+3*m; j+=3){
			double a = x[j+1];
			double k = x[j+2];

			double En = 0.0;
			double Ea = 1.0/m*sin(2*PI*k+phi_e)*(Jn_(a) + a*Jn_2(a));
			double Ek = 1.0/m*a*Jn_(a)*cos(2*PI*k+phi_e)*2*PI;

			MatSetValue(B, 0, j,   -En, INSERT_VALUES);
			MatSetValue(B, 0, j+1, -Ea, INSERT_VALUES);
			MatSetValue(B, 0, j+2, -Ek, INSERT_VALUES);
		}

		double phie = delta_e;
		double phiphi = -1.0/m * sum_sin;

		MatSetValue(B, 1, 0, -phie, INSERT_VALUES);
		MatSetValue(B, 1, 1, shift-phiphi, INSERT_VALUES);

		for(int j = 2; j<2+3*m; j+=3){
			double a = x[j+1];
			double k = x[j+2];

			double phin = 0.0;
			double phia = 1.0/m*cos(2*PI*k+phi_e)*(Jn_(a) + a*Jn_2(a));
			double phik = -1.0/m*a*Jn_(a)*sin(2*PI*k+phi_e)*2*PI;

			MatSetValue(B, 1, j,   -phin, INSERT_VALUES);
			MatSetValue(B, 1, j+1, -phia, INSERT_VALUES);
			MatSetValue(B, 1, j+2, -phik, INSERT_VALUES);
		}
	}// rank 0

	// 2 compute particles
	for(int i = lo < 2 ? 2 : lo; i<hi; i+=3){
		double a = x[i-lo+1];
		double k = x[i-lo+2];

		// n
		double ne = -r_e*a*Jn_(a)*sin(2*PI*k + phi_e);
		double nphi = -r_e*E_e*a*Jn_(a)*cos(2*PI*k + phi_e);
		double nn = 0.0;
		double na = -r_e*E_e*sin(2*PI*k + phi_e)*(Jn_(a)+a*Jn_2(a));
		double nk = -r_e*E_e*a*Jn_(a)*cos(2*PI*k + phi_e)*2*PI;

		MatSetValue(B, i, 0, -ne, INSERT_VALUES);
		MatSetValue(B, i, 1, -nphi, INSERT_VALUES);

		MatSetValue(B, i, i,  shift-nn, INSERT_VALUES);
		MatSetValue(B, i, i+1,     -na, INSERT_VALUES);
		MatSetValue(B, i, i+2,     -nk, INSERT_VALUES);

		// a
		double ae = -n*Jn_(a)*sin(2*PI*k + phi_e);
		double aphi = -n*E_e*Jn_(a)*cos(2*PI*k + phi_e);
		double an = 0.0;
		double aa = -n*E_e*sin(2*PI*k + phi_e)*Jn_2(a);
		double ak = -n*E_e*Jn_(a)*cos(2*PI*k + phi_e)*2*PI;

		MatSetValue(B, i+1, 0, -ae, INSERT_VALUES);
		MatSetValue(B, i+1, 1, -aphi, INSERT_VALUES);

		MatSetValue(B, i+1, i,        -an, INSERT_VALUES);
		MatSetValue(B, i+1, i+1, shift-aa, INSERT_VALUES);
		MatSetValue(B, i+1, i+2,      -ak, INSERT_VALUES);

		// k
		double ke = n*Jn(a)*(1-n*n/a/a)*cos(2*PI*k + phi_e) / 2.0 / PI;
		double kphi = -n*E_e*Jn(a)*(1-n*n/a/a)*sin(2*PI*k + phi_e) / 2.0 / PI;
		double kn = 1.0 / 2.0 / PI;
		double ka = n*gamma_0_2*r_e*a / 2.0 / PI - n*E_e*cos(2*PI*k + phi_e)*(2*Jn(a)/a/a/a + Jn_(a)*(1-n*n/a/a)) / 2.0 / PI;
		double kk = -n*E_e*Jn(a)*(1-n*n/a/a)*sin(2*PI*k + phi_e);//2*PI / 2.0 / PI;

		MatSetValue(B, i+2, 0, -ke, INSERT_VALUES);
		MatSetValue(B, i+2, 1, -kphi, INSERT_VALUES);

		MatSetValue(B, i+2, i,        -kn, INSERT_VALUES);
		MatSetValue(B, i+2, i+1,      -ka, INSERT_VALUES);
		MatSetValue(B, i+2, i+2, shift-kk, INSERT_VALUES);
	}

	}// if use_ijacobian

	VecRestoreArray(X, &x);
	VecRestoreArray(Xdot, &xdot);

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	if (A != B) {
		MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
	}
	return 0;
}

PetscErrorCode RHSFunction_tm(TS ts, PetscReal t,Vec in,Vec out,void*){
//	fprintf(stderr, "%s\n", __FUNCTION__);
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
	const double* data;
	ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
	if(rank == 0){
		E_e = data[0];
		phi_e = data[1];
	}
	ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
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
//		double dE = theta_e*E_e + 2.0/m*sum_cos;
//		double dphi = (delta_e*E_e - 2.0/m*sum_sin) / E_e;
			VecSetValue(out, 0, dE, INSERT_VALUES);
			VecSetValue(out, 1, dphi, INSERT_VALUES);
	}
	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

	// compute n, a, k
	VecGetOwnershipRange(out, &lo, &hi);
	if(lo < 2)
		lo = 2;

	for(int i=lo; i<hi; i+=3){
		double nak[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, nak);

//		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1]) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);
		double da = -n/nak[1]*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
		double dn = -r_e*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e) - alpha*da;

		//previous version - before simplifications - corresponds to Jacobians!!
//		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1]) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);

		// simplified version - contradicts to jacobians!!
		double dk = nak[0]*(1-gamma_0_2) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);

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

PetscErrorCode ifunction_tm(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	PetscScalar    *x,*xdot,*f;

	VecGetArray(X,&x);
	VecGetArray(Xdot,&xdot);
	VecGetArray(F,&f);

	double E_e, phi_e;

	// bcast E and phi
	if(rank == 0){
		E_e = x[0];
		phi_e = x[1];
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(X, &lo, &hi);

	double sum_sin = 0;
	double sum_cos = 0;
	for(int i=lo < 2? 2 : lo; i<hi; i+=3){
		double nak[3] = {x[i-lo], x[i+1-lo], x[i+2-lo]};
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
	double dE, dphi;
	if(rank == 0){
		dE = theta_e*E_e + 1.0/m*sum_cos;
		dphi = ( delta_e*E_e - 1.0/m*sum_sin ) / E_e;
		f[0] = xdot[0] - dE;
		f[1] = xdot[1] - dphi;
	}
	MPI_Bcast(&dE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dphi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// compute n, a, k
	for(int i=lo<2? 2 : lo; i<hi; i+=3){
		double nak[3] = {x[i-lo], x[i+1-lo], x[i+2-lo]};

		double dn = -r_e*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
		double da = -n/nak[1]*E_e*Jn(nak[1])*cos(2*PI*nak[2]+phi_e);
		double dk = nak[0] + 0.5*n*gamma_0_2*r_e*(a0*a0 - nak[1]*nak[1]) + n*E_e/nak[1]*Jn_(nak[1])*sin(2*PI*nak[2]+phi_e);

		dk /= 2*PI;

		f[i-lo] = xdot[i-lo] - dn;
		f[i+1-lo] = xdot[i+1-lo] - da;
		f[i+2-lo] = xdot[i+2-lo] - dk;
	}

	VecRestoreArray(X,&x);
	VecRestoreArray(Xdot,&xdot);
	VecRestoreArray(F,&f);

	return 0;
}

PetscErrorCode ijacobian_tm(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal shift, Mat A, Mat B,void *ctx){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	PetscScalar *x, *xdot;

	int lo, hi;
	MatGetOwnershipRange(B, &lo, &hi);

	VecGetArray(X, &x);
	VecGetArray(Xdot, &xdot);

	// 0 prepare
	double E_e, phi_e;
	double sum_sin = 0, sum_cos = 0;

	// bcast E and phi
	if(lo == 0){
		E_e = x[0];
		phi_e = x[1];
	}
	MPI_Bcast(&E_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi_e, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	for(int i = lo < 2 ? 2 : lo; i<hi; i+=3){
		double a = x[i-lo+1];
		double k = x[i-lo+2];
		sum_sin += Jn(a)*sin(2*PI*k+phi_e);
		sum_cos += Jn(a)*cos(2*PI*k+phi_e);
	}
	double tmp = 0;
	MPI_Reduce(&sum_sin, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_sin = tmp;
	tmp = 0;
	MPI_Reduce(&sum_cos, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_cos = tmp;

	// 1 add a-term
	for(int i=lo; i<hi; i++){
		MatSetValue(B, i, i, shift, INSERT_VALUES);
	}

	if(use_ijacobian){

	// 2 compute E and phi
	if(lo==0){
		double Ee = theta_e;
		double Ephi = -1.0/m * sum_sin;

		MatSetValue(B, 0, 0, shift-Ee, INSERT_VALUES);
		MatSetValue(B, 0, 1, -Ephi, INSERT_VALUES);

		for(int j = 2; j<2+3*m; j+=3){
			double a = x[j+1];
			double k = x[j+2];

			double En = 0.0;
			double Ea = 1.0/m*Jn_(a)*cos(2*PI*k+phi_e);
			double Ek = -1.0/m*Jn(a)*sin(2*PI*k+phi_e)*2*PI;

			MatSetValue(B, 0, j,   -En, INSERT_VALUES);
			MatSetValue(B, 0, j+1, -Ea, INSERT_VALUES);
			MatSetValue(B, 0, j+2, -Ek, INSERT_VALUES);
		}

		double phie = delta_e;
		double phiphi = -1.0/m * sum_cos;

		MatSetValue(B, 1, 0, -phie, INSERT_VALUES);
		MatSetValue(B, 1, 1, shift-phiphi, INSERT_VALUES);

		for(int j = 2; j<2+3*m; j+=3){
			double a = x[j+1];
			double k = x[j+2];

			double phin = 0.0;
			double phia = -1.0/m*Jn_(a)*sin(2*PI*k+phi_e);
			double phik = -1.0/m*Jn(a)*cos(2*PI*k+phi_e)*2*PI;

			MatSetValue(B, 1, j,   -phin, INSERT_VALUES);
			MatSetValue(B, 1, j+1, -phia, INSERT_VALUES);
			MatSetValue(B, 1, j+2, -phik, INSERT_VALUES);
		}
	}// rank 0

	// 2 compute particles
	for(int i = lo < 2 ? 2 : lo; i<hi; i+=3){
		double a = x[i-lo+1];
		double k = x[i-lo+2];

		// n
		double ne = -r_e*Jn(a)*cos(2*PI*k + phi_e);
		double nphi = r_e*E_e*Jn(a)*sin(2*PI*k + phi_e);
		double nn = 0.0;
		double na = -r_e*E_e*Jn_(a)*cos(2*PI*k + phi_e);
		double nk = r_e*E_e*Jn(a)*sin(2*PI*k + phi_e)*2*PI;

		MatSetValue(B, i, 0, -ne, INSERT_VALUES);
		MatSetValue(B, i, 1, -nphi, INSERT_VALUES);

		MatSetValue(B, i, i,  shift-nn, INSERT_VALUES);
		MatSetValue(B, i, i+1,     -na, INSERT_VALUES);
		MatSetValue(B, i, i+2,     -nk, INSERT_VALUES);

		// a
		double ae = -n/a*Jn(a)*cos(2*PI*k + phi_e);
		double aphi = n*E_e/a*Jn(a)*sin(2*PI*k + phi_e);
		double an = 0.0;
		double aa = -n*E_e*cos(2*PI*k + phi_e)*(Jn_(a)*a-Jn(a))/a/a;
		double ak = n*E_e/a*Jn(a)*sin(2*PI*k + phi_e)*2*PI;

		MatSetValue(B, i+1, 0, -ae, INSERT_VALUES);
		MatSetValue(B, i+1, 1, -aphi, INSERT_VALUES);

		MatSetValue(B, i+1, i,        -an, INSERT_VALUES);
		MatSetValue(B, i+1, i+1, shift-aa, INSERT_VALUES);
		MatSetValue(B, i+1, i+2,      -ak, INSERT_VALUES);

		// k
		double ke = n/a*Jn_(a)*sin(2*PI*k + phi_e) / 2.0 / PI;
		double kphi = n*E_e/a*Jn_(a)*cos(2*PI*k + phi_e) / 2.0 / PI;
		double kn = 1.0 / 2.0 / PI;
		double ka = -n*gamma_0_2*r_e*a / 2.0 / PI + n*E_e*sin(2*PI*k + phi_e)*(Jn_2(a)*a - Jn_(a))/a/a / 2.0 / PI;
		double kk = n*E_e/a*Jn_(a)*cos(2*PI*k + phi_e);//2*PI / 2.0 / PI;

		MatSetValue(B, i+2, 0, -ke, INSERT_VALUES);
		MatSetValue(B, i+2, 1, -kphi, INSERT_VALUES);

		MatSetValue(B, i+2, i,        -kn, INSERT_VALUES);
		MatSetValue(B, i+2, i+1,      -ka, INSERT_VALUES);
		MatSetValue(B, i+2, i+2, shift-kk, INSERT_VALUES);
	}

	}// if use_ijacobian

	VecRestoreArray(X, &x);
	VecRestoreArray(Xdot, &xdot);

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	if (A != B) {
		MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
	}
	return 0;
}

// turned it off because PETSc 5.6 does not allow to change vector in step
// now wrap is done in vec_to_state
void wrap_ksi_in_vec(Vec u){
	int size;
	VecGetLocalSize(u, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	double* arr;
	VecGetArray(u, &arr);

	int i;			// point at ksi
	if(rank==0)
		i = 4;
	else
		i= 2;

	for(; i<size; i+=3){
		//fprintf(stderr, "%lf:", arr[i]);
		if(arr[i] > 0.5)
			arr[i] -= 1.0;
		else if(arr[i] < -0.5)
			arr[i] += 1.0;
		//fprintf(stderr, "%lf ", arr[i]);
	}
	//fprintf(stderr, "\n");
	VecRestoreArray(u, &arr);
}

// NOTE: this doesn't work in TSPostStep - because solution is rewritten after call to this func
// use it in TSMonitor
PetscErrorCode wrap_function(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx){
//	Vec u;
//	TSGetSolution(ts, &u);
//	wrap_ksi_in_vec(u);

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

