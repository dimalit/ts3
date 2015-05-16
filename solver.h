/*
 * solver.h
 *
 *  Created on: Feb 19, 2015
 *      Author: dimalit
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscts.h>
#include <mpi.h>

extern double init_step;
extern double atolerance, rtolerance;
extern int m;
extern double n;
extern double theta_e, delta_e, r_e, gamma_0_2;
extern double a0;
extern bool use_ifunction, use_ijacobian;

// both out vectors must be destroyed outside
PetscErrorCode solve_te(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time);

PetscErrorCode solve_tm(Vec initial_state, int max_steps, double max_time,
		   Vec* out_state, Vec* out_rhs, int* out_steps, double* out_time);

#endif /* SOLVER_H_ */
