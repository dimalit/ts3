/*
 * main.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: dimalit
 */

#include <model_e3.pb.h>
using namespace pb;

#include "solver.h"

#include <petscts.h>
#include <mpi.h>
#include <unistd.h>

E3Config pconfig;
E3PetscSolverConfig sconfig;
E3State state;

int rank;
int size;

void vec_to_state(Vec v, E3State*);
void state_to_vec(const E3State* state, Vec v);

// TMP
//#include <fcntl.h>

void broadcast_message(google::protobuf::Message& msg){
	char* buf; int buf_size;

	if(rank == 0)
		buf_size = msg.ByteSize();
	MPI_Bcast(&buf_size, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	buf = new char[buf_size];

	if(rank == 0)
		msg.SerializeToArray(buf, buf_size);

	MPI_Bcast(buf, buf_size, MPI_BYTE, 0, PETSC_COMM_WORLD);

	if(rank != 0)
		msg.ParseFromArray(buf, buf_size);

	delete[] buf;
}

int main(int argc, char** argv){

//	int go = 0;
//	while(go==0){
//		sleep(1);
//	}

//	close(0);
//	open("../ode-env/all.tmp", O_RDONLY);

	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char*)0, (char*)0);CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

	int max_steps; double max_time;

	if(rank==0){
		E3Model all;
		all.ParseFromFileDescriptor(0);

		max_steps = all.steps();
		max_time = all.time();

		sconfig.CopyFrom(all.sconfig());
		pconfig.CopyFrom(all.pconfig());
		state.CopyFrom(all.state());
	}

	MPI_Bcast(&max_steps, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&max_time, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	broadcast_message(sconfig);
	broadcast_message(pconfig);

	// set global parameters
	init_step = sconfig.init_step();
	atolerance = sconfig.atol();
	rtolerance = sconfig.rtol();
	m = pconfig.m();
	n = pconfig.n();
	theta_e = pconfig.theta_e();
	delta_e = pconfig.delta_e();
	r_e = pconfig.r_e();
	gamma_0_2 = pconfig.gamma_0_2();
	a0 = state.a0();

	use_ifunction = false;
	use_ijacobian = false;
	if(sconfig.solver()!=sconfig.rhs)
		use_ifunction = true;
	if(sconfig.solver()==sconfig.ijacobian)
		use_ijacobian = true;

	Vec u;
	VecCreate(PETSC_COMM_WORLD, &u);
	VecSetType(u, VECMPI);
	VecSetSizes(u, PETSC_DECIDE, pconfig.m()*3+2);
	state_to_vec(&state, u);

	Vec res, res_rhs;
	int res_steps;
	double res_time;

	if(sconfig.model() == "te")
		solve_te(u, max_steps, max_time, &res, &res_rhs, &res_steps, &res_time);
	else if(sconfig.model() == "tm")
		solve_tm(u, max_steps, max_time, &res, &res_rhs, &res_steps, &res_time);
	else{
		fprintf(stderr, "assert");
		fflush(stderr);
		assert(false);
	}

	//void wrap_ksi_in_vec(Vec u);
	//wrap_ksi_in_vec(res);

	E3Solution sol;
	for(int i=0; i<m; i++){
		sol.mutable_state()->add_particles();
		sol.mutable_d_state()->add_particles();
	}

	vec_to_state(res, sol.mutable_state());
	vec_to_state(res_rhs, sol.mutable_d_state());
	sol.mutable_state()->set_a0(a0);
	sol.mutable_d_state()->set_a0(0);

	if(rank == 0){
		// 1 write time and steps
		write(1, &res_steps, sizeof(res_steps));
		write(1, &res_time, sizeof(res_time));

		// 2 write state
		sol.SerializeToFileDescriptor(1);
	}

	VecDestroy(&u);
	VecDestroy(&res);
	VecDestroy(&res_rhs);

	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}

void state_to_vec(const E3State* state, Vec v){
	int vecsize;
	VecGetSize(v, &vecsize);
	assert(vecsize == pconfig.m()*3+2);

	// 1 write E and phi
	if(rank == 0){
		VecSetValue(v, 0, state->e(), INSERT_VALUES);
		VecSetValue(v, 1, state->phi(), INSERT_VALUES);
	}
	VecAssemblyBegin(v);
	VecAssemblyEnd(v);

	// 2 scatter data!
	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders = new PetscInt[size+1];
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){
		for(int r = size-1; r>=0; r--){		// go down - because last will be mine
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += 2;

			assert((lo-2)%3 == 0);
			assert((hi-2)%3 == 0);

			int first = (lo-2) / 3;
			int count = (hi - lo) / 3;

			int add = r==0 ? 2 : 0;
			for(int i=0; i<count; i++){
				E3State::Particles p = state->particles(first+i);
				arr[add + i*3+0] = p.eta();
				arr[add + i*3+1] = p.a();
				arr[add + i*3+2] = p.ksi();
			}

			if(r!=0)
				MPI_Send(arr, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD);

		}// for
	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		MPI_Status s;
		assert(MPI_SUCCESS == MPI_Recv(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &s));
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

void vec_to_state(Vec v, E3State* state){
	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders = new PetscInt[size+1];
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		state->set_e(arr[0]);
		state->set_phi(arr[1]);

		for(int r = 0; r<size; r++){
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += 2;

			assert((lo-2)%3 == 0);
			assert((hi-2)%3 == 0);

			int first = (lo-2) / 3;
			int count = (hi - lo) / 3;

			MPI_Status s;
			if(r!=0)
				assert(MPI_SUCCESS == MPI_Recv(arr, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD, &s));

			int add = r==0 ? 2 : 0;
			for(int i=0; i<count; i++){
				E3State::Particles* p = state->mutable_particles(first+i);
				p->set_eta(arr[add + i*3+0]);
				p->set_a(arr[add + i*3+1]);
				p->set_ksi(arr[add + i*3+2]);
			}
		}// for
	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		assert(MPI_SUCCESS == MPI_Send(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD));
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

